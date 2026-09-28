"""检查已有适应曲线、固定私有模块时的共享权重贡献和目标表示。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.load import load_scans, support_data
from data.preprocess import Preprocessor
from evaluation.metrics import evaluate
from models.femloc import Localizer, private_modules, shared_module


def vector(state):
    return torch.cat([v.flatten() for v in state.values()])


def main():
    setup = yaml.safe_load(Path('configs/diagnostics.yaml').read_text())
    config = yaml.safe_load(Path('configs/v1.yaml').read_text())
    torch.set_num_threads(config['torch_threads'])
    out = Path(setup['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'diagnostics.yaml').write_text(yaml.safe_dump(setup))
    frame, audit = load_scans(config['data_path'], config)
    curves, swaps, representations, fitting, displacements = [], [], [], [], []
    for root in sorted(Path('outputs/full').iterdir()):
        run = json.loads((root / 'run.json').read_text())
        manifest = json.loads(Path(run['manifest']).read_text())
        assert manifest['audit'] == audit
        key = {k: run[k] for k in ['task', 'target', 'repeat']}
        domains = [s['domain'] for s in manifest['sources']]
        source = torch.load(root / 'source_shared.pt', map_location=config['device'], weights_only=True)
        if not run.get('source_reused_from'):
            a, b = vector(source['initial']), vector(source['shared'])
            displacements.append({**key, 'relative_change': float((b-a).norm()/a.norm())})
        curve = pd.read_csv(root / 'metrics.csv')
        curves.append(curve[curve.method.isin(['FeMLoc', 'FeMLoc-RI'])])
        for method in ['FeMLoc', 'FeMLoc-RI']:
            saved = torch.load(root / f'{method}.pt', map_location=config['device'], weights_only=True)
            prep = Preprocessor(**saved['preprocessor'])
            prep.ap_indices = np.asarray(prep.ap_indices)
            prep.origin = np.asarray(prep.origin)
            encoder, _, mapper = private_modules(len(prep.ap_indices), config, run['target'], run['repeat'])
            model = Localizer(encoder, shared_module(config, domains, run['repeat']), mapper).eval()
            datasets = {split: support_data(frame, manifest[f'target_{split}']) for split in ['support', 'query']}
            for step in setup['representation_steps']:
                model.load_state_dict(saved['snapshots'][step])
                for split, data in datasets.items():
                    x = prep.x(data['rssi'], compact=True, device=config['device'])
                    with torch.no_grad():
                        z = model.encoder(x)
                        prediction = prep.inverse(model(x))
                    stats, _ = evaluate(frame, data['row_ids'], prediction)
                    fitting.append({**key, 'method': method, 'step': step, 'split': split, **stats})
                    singular = torch.linalg.svdvals(z-z.mean(0))
                    energy = singular.square()
                    rank = float(energy.sum().square()/energy.square().sum()) if energy.sum() > 0 else 0.
                    representations.append({**key, 'method': method, 'step': step, 'split': split,
                        'latent_rms': float(z.square().mean().sqrt()),
                        'latent_centered_rms': float((z-z.mean(0)).square().mean().sqrt()),
                        'effective_rank': rank})
                    if step == 0 and method == 'FeMLoc':
                        results = {}
                        for name in ['initial', 'shared']:
                            model.shared.load_state_dict(source[name])
                            with torch.no_grad():
                                pred = prep.inverse(model(x))
                            metric, _ = evaluate(frame, data['row_ids'], pred)
                            results[name] = (pred, metric['position_mean'])
                        shift = np.linalg.norm(results['shared'][0]-results['initial'][0], axis=1).mean()
                        swaps.append({**key, 'split': split, 'mean_prediction_shift': shift,
                            'error_initial': results['initial'][1], 'error_trained': results['shared'][1]})
                        model.load_state_dict(saved['snapshots'][step])
        print(root.name, flush=True)
    pd.concat(curves).to_csv(out / 'curves.csv', index=False)
    for name, rows in [('fixed_private_swap', swaps), ('representations', representations),
                       ('fit_generalization', fitting), ('source_displacement', displacements)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    (out / 'audit_completed.json').write_text(json.dumps({'groups': len(curves), 'audit': audit}))


if __name__ == '__main__':
    main()
