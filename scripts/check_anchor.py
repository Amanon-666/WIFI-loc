"""检查参考点标签隔离、张量接口和一阶元更新。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.anchors import ScanStore, episode
from data.load import load_scans
from models.anchor import make_network
from training.anchor import source_train


def main():
    torch.set_num_threads(1)
    c = yaml.safe_load(Path('configs/v1.yaml').read_text())
    c['anchor'] = yaml.safe_load(Path('configs/anchor_v2.yaml').read_text())['anchor']
    frame, _ = load_scans(c['data_path'], c)
    store = ScanStore(frame, c)
    m = json.loads(Path('outputs/manifests/T1_B0F1_e0.json').read_text())
    sx, sy, qx, qy, _ = episode(store, m['target_support'], m['target_query'], 'anchor')
    assert sx.shape == (30, 15) and qx.shape[1] == 15 and sy.shape == (30, 2)
    changed = deepcopy(store)
    ids = [store.offset[row] for row in m['target_support']]
    same = (store.xy[ids] == store.xy[ids[0]]).all(1)
    changed.xy[np.asarray(ids)[same]] += [500, 300]
    sx2, _, _, _, _ = episode(changed, m['target_support'], [], 'anchor')
    np.testing.assert_array_equal(sx.numpy()[same], sx2.numpy()[same])
    changed = deepcopy(store)
    changed.xy[[store.offset[row] for row in m['target_query']]] += 10000
    _, _, qx2, _, _ = episode(changed, m['target_support'], m['target_query'], 'anchor')
    np.testing.assert_array_equal(qx.numpy(), qx2.numpy())
    model = make_network(15, c['anchor']['hidden'], c['seed'])
    before = torch.cat([p.detach().flatten() for p in model.parameters()])
    c['anchor']['rounds'] = 1
    source_train(model, store, m['sources'][:1], c, 'anchor', 'maml')
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    assert torch.isfinite(after).all() and not torch.equal(before, after)
    print('PASS: self-position exclusion, query-label isolation, shapes and meta update')


if __name__ == '__main__':
    main()
