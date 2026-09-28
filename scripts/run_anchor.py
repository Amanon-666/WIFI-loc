"""运行固定 Manifest 上的关系表示和元学习对照，保存逐步预测。"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

from data.anchors import ScanStore, episode, query_input
from data.load import load_scans, query_features, support_data
from data.preprocess import Preprocessor
from evaluation.metrics import evaluate
from models.anchor import make_network
from models.wknn import predict_wknn
from training.anchor import source_train, target_adapt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/anchor_v2.yaml')
    args = parser.parse_args()
    base = yaml.safe_load(Path('configs/v1.yaml').read_text())
    setup = yaml.safe_load(Path(args.config).read_text())
    config = deepcopy(base)
    config['anchor'] = setup['anchor']
    torch.set_num_threads(1)
    frame, audit = load_scans(base['data_path'], base)
    store = ScanStore(frame, base)
    for target in setup['targets']:
        for repeat in setup['repeats']:
            manifest = json.loads(Path(f'outputs/manifests/T1_{target}_e{repeat}.json').read_text())
            source_manifest = json.loads(Path(f'outputs/manifests/T2_B2F3_e{repeat}.json').read_text())
            manifest['sources'] = [s for s in source_manifest['sources'] if s['domain'] != target and int(s['domain'][1]) in setup['source_buildings']]
            manifest['protocol'] = 'UJI-Anchor-v2-development'
            assert manifest['audit'] == audit
            assert not set(frame.loc[manifest['target_support'], 'position']) & set(frame.loc[manifest['target_query'], 'position'])
            out = Path(setup['output'])/f'{target}_e{repeat}'
            out.mkdir(parents=True, exist_ok=False)
            (out/'manifest.json').write_text(json.dumps(manifest))
            (out/'config.yaml').write_text(yaml.safe_dump({**config, **setup}))
            support = support_data(frame, manifest['target_support'])
            query = query_features(frame, manifest['target_query'])
            records = []
            prediction = predict_wknn(support, query, Preprocessor.fit(support, base), base)
            stats, table = evaluate(frame, manifest['target_query'], prediction)
            records.append(dict(method='WKNN', step=0, target=target, repeat=repeat, **stats))
            table.to_csv(out/'WKNN_predictions.csv', index=False)
            for representation in setup['representations']:
                sx, sy, _, _, _ = episode(store, manifest['target_support'], [], representation, config['anchor']['neighbors'])
                started = perf_counter()
                qx, origin = query_input(store, manifest['target_support'], query['rssi'], config, representation)
                preprocessing_seconds = perf_counter()-started
                initial = make_network(sx.shape[1], config['anchor']['hidden'], base['seed']+repeat,
                                       residual=representation == 'scan')
                for algorithm in ['scratch', 'erm', 'maml']:
                    model = deepcopy(initial)
                    name = representation+'-'+algorithm
                    training_seconds = 0
                    if algorithm != 'scratch':
                        log, training_seconds = source_train(model, store, manifest['sources'], config, representation, algorithm)
                        pd.DataFrame(log).to_csv(out/f'{name}_source.csv', index=False)
                    states, losses, times = target_adapt(model, sx, sy, config)
                    torch.save({'source': model.state_dict(), 'snapshots': states, 'support_losses': losses}, out/f'{name}.pt')
                    for step, state in states.items():
                        model.load_state_dict(state)
                        with torch.no_grad():
                            prediction = origin+100*model(qx).numpy().astype('float64')
                        stats, table = evaluate(frame, manifest['target_query'], prediction)
                        records.append(dict(method=name, step=step, target=target, repeat=repeat,
                                            source_seconds=training_seconds, adapt_seconds=times[step],
                                            query_preprocess_seconds=preprocessing_seconds, **stats))
                        table.to_csv(out/f'{name}_{step}_predictions.csv', index=False)
                    pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
                    print(target, repeat, name, '20/500 m:', [round(r['position_mean'], 3) for r in records if r['method'] == name and r['step'] in [20, 500]], flush=True)
            assert np.isfinite([r['position_mean'] for r in records]).all()
            (out/'completed.json').write_text(json.dumps({'target': target, 'repeat': repeat, 'methods': sorted(set(r['method'] for r in records))}))


if __name__ == '__main__':
    main()
