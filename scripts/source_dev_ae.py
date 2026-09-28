"""在源建筑内部留出楼层，对照 AE 修正对定位适应的影响。"""
import json
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.load import load_scans, support_data, query_features
from scripts.run import save_run
from training.adapt import prepare_target, adapt
from training.source import meta_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/source_dev_ae.yaml')
    args = parser.parse_args()
    setup = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path('configs/v1.yaml').read_text())
    torch.set_num_threads(base['torch_threads'])
    source = json.loads(Path(setup['source_manifest']).read_text())
    dev = json.loads(Path(setup['development_manifest']).read_text())
    batches = json.loads(Path('outputs/diagnostics_v1/ae_followup_batches.json').read_text())
    frame, audit = load_scans(base['data_path'], base)
    assert audit == source['audit'] == dev['audit']
    manifest = deepcopy(dev)
    manifest['task'] = 'source-development'
    manifest['sources'] = [deepcopy(s) for s in source['sources'] if s['domain'] != dev['target']]
    assert len(manifest['sources']) == 7
    assert all(s['domain'][:2] != 'B2' for s in manifest['sources']) and dev['target'][:2] != 'B2'
    for variant, ae in setup['variants'].items():
        out = Path(setup['output'])/variant
        out.mkdir(parents=True, exist_ok=False)
        config = deepcopy(base)
        config['ae'].update({k: ae[k] for k in ['lr', 'steps']})
        if 'outer_lr' in ae:
            config['meta']['outer_lr'] = ae['outer_lr']
        for item in manifest['sources']:
            item['ae_batches'] = batches[item['domain']][:ae['steps']]
        (out/'manifest.json').write_text(json.dumps(manifest))
        (out/'config.yaml').write_text(yaml.safe_dump(config))
        shared, initial, clients, log = meta_train(frame, manifest, config, config['meta']['rounds'])
        del clients
        torch.save({'shared': shared.state_dict(), 'initial': initial.state_dict()}, out/'source_shared.pt')
        (out/'source_training.json').write_text(json.dumps(log))
        support = support_data(frame, manifest['target_support'])
        features = query_features(frame, manifest['target_query'])
        private = prepare_target(support, manifest['target'], manifest['repeat'], config)
        rows = []
        common = {'task': manifest['task'], 'target': manifest['target'], 'repeat': manifest['repeat'],
                  'variant': variant, 'source_rounds': config['meta']['rounds']}
        for name, state in [('FeMLoc-RI', initial), ('FeMLoc', shared)]:
            result = adapt(state, support, private, config)
            rows += save_run(name, result, features, frame, out, config, common)
        pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
        (out/'completed.json').write_text(json.dumps(common))
        print('Development comparison complete:', variant, flush=True)


if __name__ == '__main__':
    main()
