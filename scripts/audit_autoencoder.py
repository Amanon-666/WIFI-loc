"""核验目标 AE 输出饱和，并仅在源楼层对照 AE 学习率。"""
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.load import load_scans, support_data
from data.preprocess import Preprocessor
from models.femloc import private_modules
from training.common import train_ae


def measure(encoder, decoder, x, constant):
    with torch.no_grad():
        z = encoder(x)
        pred = decoder(z)
        return {'mse': float((pred-x).square().mean()),
                'constant_mse': float((constant-x).square().mean()),
                'zero_mse': float(x.square().mean()),
                'output_max': float(pred.max()),
                'output_below_1e6_fraction': float((pred < 1e-6).float().mean()),
                'latent_rms': float(z.square().mean().sqrt())}


def main():
    setup = yaml.safe_load(Path('configs/diagnostics.yaml').read_text())
    config = yaml.safe_load(Path('configs/v1.yaml').read_text())
    torch.set_num_threads(config['torch_threads'])
    frame, _ = load_scans(config['data_path'], config)
    records = []
    for name in setup['source_replays']:
        m = json.loads(Path(f'outputs/manifests/{name}.json').read_text())
        data = support_data(frame, m['target_support'])
        prep = Preprocessor.fit(data, config)
        x = prep.x(data['rssi'], compact=True, device=config['device'])
        e, d, _ = private_modules(len(prep.ap_indices), config, m['target'], m['repeat'])
        train_ae(e, d, x, [torch.arange(len(x), device=x.device)]*config['ae']['steps'], config)
        records.append({'role': 'target_audit_only', 'domain': m['target'], 'split': 'support',
                        'lr': config['ae']['lr'], **measure(e, d, x, x.mean(0))})
    m = json.loads(Path('outputs/manifests/T2_B2F3_e0.json').read_text())
    assert all(item['domain'][:2] != 'B2' for item in m['sources'])
    for item in m['sources']:
        train = support_data(frame, item['fit_rows'])
        prep = Preprocessor.fit(train, config)
        x = prep.x(train['rssi'], compact=True, device=config['device'])
        query = support_data(frame, item['query_rows'])
        q = prep.x(query['rssi'], compact=True, device=config['device'])
        offset = {row: i for i, row in enumerate(train['row_ids'])}
        batches = [torch.tensor([offset[r] for r in rows], device=x.device) for rows in item['ae_batches']]
        for lr in setup['ae_learning_rates']:
            c = deepcopy(config)
            c['ae']['lr'] = lr
            e, d, _ = private_modules(len(prep.ap_indices), config, item['domain'], m['repeat'])
            train_ae(e, d, x, batches, c)
            for split, inputs in [('source_pc', x), ('source_pq', q)]:
                records.append({'role': 'source_lr_diagnostic', 'domain': item['domain'], 'split': split,
                                'lr': lr, **measure(e, d, inputs, x.mean(0))})
            print(item['domain'], lr, flush=True)
    out = Path(setup['output'])
    pd.DataFrame(records).to_csv(out/'ae_saturation_and_rates.csv', index=False)
    (out/'ae_completed.json').write_text(json.dumps({'source_domains': [s['domain'] for s in m['sources']],
                                                   'learning_rates': setup['ae_learning_rates']}))


if __name__ == '__main__':
    main()
