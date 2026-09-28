"""在源楼层固定较低学习率，检查 AE 重建随训练步数的变化。"""
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.load import load_scans, support_data
from data.preprocess import Preprocessor
from models.femloc import private_modules
from scripts.audit_autoencoder import measure
from tasks.build import position_index, partition, uniform_batches
from training.common import adam, supervised_step


def main():
    setup = yaml.safe_load(Path('configs/diagnostics.yaml').read_text())
    config = yaml.safe_load(Path('configs/v1.yaml').read_text())
    torch.set_num_threads(config['torch_threads'])
    frame, _ = load_scans(config['data_path'], config)
    manifest = json.loads(Path('outputs/manifests/T2_B2F3_e0.json').read_text())
    output = Path(setup['output'])
    records, saved_batches = [], {}
    for item in manifest['sources']:
        domain = item['domain']
        assert domain[:2] != 'B2'
        index = position_index(frame, domain)
        pc, _ = partition(index, config, domain)
        batches = uniform_batches(index, pc, config, domain, 0, 'ae_sampling',
                                  max(setup['ae_followup_steps']), config['ae']['source_batch_size'])
        assert batches[:config['ae']['steps']] == item['ae_batches']
        saved_batches[domain] = batches
        train = support_data(frame, item['fit_rows'])
        prep = Preprocessor.fit(train, config)
        x = prep.x(train['rssi'], True, config['device'])
        q = prep.x(support_data(frame, item['query_rows'])['rssi'], True, config['device'])
        offsets = {r: i for i, r in enumerate(train['row_ids'])}
        e, d, _ = private_modules(len(prep.ap_indices), config, domain, 0)
        model = torch.nn.Sequential(e, d)
        optimizer = adam(model.parameters(), config, setup['ae_followup_lr'])
        for step, rows in enumerate(batches, 1):
            inputs = x[torch.tensor([offsets[r] for r in rows], device=x.device)]
            supervised_step(model, inputs, inputs, [optimizer])
            if step in setup['ae_followup_steps']:
                for split, inputs in [('source_pc', x), ('source_pq', q)]:
                    records.append({'domain': domain, 'step': step, 'split': split,
                                    **measure(e, d, inputs, x.mean(0))})
        print(domain, 'AE schedule complete', flush=True)
    (output/'ae_followup_batches.json').write_text(json.dumps(saved_batches))
    pd.DataFrame(records).to_csv(output/'ae_schedule.csv', index=False)
    (output/'ae_schedule_completed.json').write_text(json.dumps({'domains': list(saved_batches),
        'lr': setup['ae_followup_lr'], 'steps': setup['ae_followup_steps']}))


if __name__ == '__main__':
    main()
