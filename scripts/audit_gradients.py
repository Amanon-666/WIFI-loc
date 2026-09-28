"""沿用 V1 源训练函数，重放短程训练并记录梯度及 AE 表示。"""
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml

from data.load import load_scans, support_data
from data.preprocess import Preprocessor
from models.femloc import shared_module, private_modules
from training.common import train_ae
from training.source import SourceClient, meta_round


def flatten(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


def main():
    setup = yaml.safe_load(Path('configs/diagnostics.yaml').read_text())
    config = yaml.safe_load(Path('configs/v1.yaml').read_text())
    torch.set_num_threads(config['torch_threads'])
    out = Path(setup['output'])
    frame, audit = load_scans(config['data_path'], config)
    records, ae, contribution, target_ae = [], [], [], []
    original_grad = torch.autograd.grad
    for name in setup['source_replays']:
        manifest = json.loads(Path(f'outputs/manifests/{name}.json').read_text())
        assert audit == manifest['audit']
        domains = [s['domain'] for s in manifest['sources']]
        support = support_data(frame, manifest['target_support'])
        prep = Preprocessor.fit(support, config)
        encoder, decoder, _ = private_modules(len(prep.ap_indices), config, manifest['target'], manifest['repeat'])
        x = prep.x(support['rssi'], compact=True, device=config['device'])
        ae_losses = train_ae(encoder, decoder, x, [torch.arange(len(x), device=x.device)]*config['ae']['steps'], config)
        saved = torch.load(Path(f'outputs/full/{name}/FeMLoc.pt'), map_location=config['device'], weights_only=True)
        assert all(torch.allclose(value, saved['snapshots'][0]['encoder.'+key], atol=1e-6, rtol=1e-5)
                   for key, value in encoder.state_dict().items())
        with torch.no_grad():
            z = encoder(x)
            target_ae.append({'replay': name, 'reconstruction_mse': float((decoder(z)-x).square().mean()),
                'zero_mse': float(x.square().mean()), 'constant_mean_mse': float((x-x.mean(0)).square().mean()),
                'latent_rms': float(z.square().mean().sqrt()), 'ae_first_mse': ae_losses[0],
                'ae_final_preupdate_mse': ae_losses[-1]})
        shared = shared_module(config, domains, manifest['repeat'])
        initial = deepcopy(shared.state_dict())
        clients = [SourceClient(frame, item, shared, manifest['repeat'], config) for item in manifest['sources']]
        for c in clients:
            idx = c.indices(c.item['fit_rows'])
            x = c.x[idx]
            ae.append({'replay': name, 'domain': c.item['domain'],
                       'ae_first_batch_mse': c.ae_losses[0], 'ae_last_batch_mse': c.ae_losses[-1],
                       'zero_reconstruction_mse_all_pc': float(x.square().mean()),
                       'constant_mean_reconstruction_mse_all_pc': float((x-x.mean(0)).square().mean())})
        for r in range(setup['source_rounds']):
            grads = []
            def capture(*args, **kwargs):
                result = original_grad(*args, **kwargs)
                grads.append(flatten(result))
                return result
            before = flatten(shared.parameters()).clone()
            with patch('torch.autograd.grad', side_effect=capture):
                losses = meta_round(shared, clients, r, config)
            assert len(grads) == len(clients)
            aggregate = torch.stack(grads).mean(0)
            delta = flatten(shared.parameters())-before
            assert torch.allclose(delta, -config['meta']['outer_lr']*aggregate, atol=2e-8, rtol=1e-3)
            records.append({'replay': name, 'round': r+1,
                'mean_client_grad_norm': float(torch.stack([g.norm() for g in grads]).mean()),
                'aggregate_grad_norm': float(aggregate.norm()),
                'gradient_alignment_ratio': float(aggregate.norm()/torch.stack([g.norm() for g in grads]).mean()),
                'relative_update': float(delta.norm()/before.norm()),
                'support_mse': sum(c['support_mse'] for c in losses)/len(clients),
                'query_mse': sum(c['query_mse'] for c in losses)/len(clients)})
        for c in clients:
            idx = c.indices(c.item['query_rows'])
            for label, state in [('initial', initial), ('trained_100', shared.state_dict())]:
                c.model.shared.load_state_dict(state)
                with torch.no_grad():
                    mse = (c.model(c.x[idx])-c.y[idx]).square().mean()
                contribution.append({'replay': name, 'domain': c.item['domain'], 'shared': label,
                                     'query_mse_fixed_private': float(mse)})
        print('Replay completed:', name, flush=True)
    pd.DataFrame(records).to_csv(out/'gradient_trace.csv', index=False)
    pd.DataFrame(ae).to_csv(out/'source_ae.csv', index=False)
    pd.DataFrame(contribution).to_csv(out/'source_fixed_private.csv', index=False)
    pd.DataFrame(target_ae).to_csv(out/'target_ae.csv', index=False)
    (out/'gradient_completed.json').write_text(json.dumps({'replays': setup['source_replays'], 'rounds': setup['source_rounds']}))


if __name__ == '__main__':
    main()
