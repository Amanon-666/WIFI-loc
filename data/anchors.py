"""将扫描转换为参考位置关系特征，或同协议的完整 RSSI 输入。"""
import numpy as np
import torch

from data.load import WAPS


class ScanStore:
    """保留行号索引，供 manifest 直接提取 RSSI 和坐标。"""
    def __init__(self, frame, config):
        self.offset = {row: i for i, row in enumerate(frame.index)}
        raw = frame[WAPS].to_numpy()
        c = config['preprocess']
        self.visible = raw != c['missing']
        self.x = np.where(self.visible, (raw-c['rssi_min'])/(c['rssi_max']-c['rssi_min']), 0).astype('float32')
        self.xy = frame[['LONGITUDE', 'LATITUDE']].to_numpy('float64')

    def rows(self, ids):
        idx = [self.offset[row] for row in ids]
        return self.x[idx], self.visible[idx], self.xy[idx]


def features(x, bank_x, bank_visible, bank_xy, neighbors=5, scans=False):
    """按 RSS 距离选不同参考位置，输出距离和中心化的参考坐标。"""
    positions, group = np.unique(bank_xy, axis=0, return_inverse=True)
    prototypes = bank_x if scans else np.stack([bank_x[group == i].mean(0) for i in range(len(positions))])
    mask = bank_visible.any(0)
    distance = np.sqrt(((x[:, None, :] - prototypes[None, :, :])**2 * mask).mean(2))
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
    origin = positions.mean(0)
    locations = bank_xy if scans else positions
    relative = (locations[nearest] - origin)/100
    values = np.concatenate([np.take_along_axis(distance, nearest, axis=1)[..., None], relative], axis=2)
    return values.reshape(len(x), neighbors*3).astype('float32'), origin


def episode(store, support_ids, query_ids, representation, neighbors=5):
    """构造内外循环张量；Support 特征逐位置排除自身参考信息。"""
    sx, sv, sy = store.rows(support_ids)
    qx, _, qy = store.rows(query_ids)
    if representation == 'raw':
        origin = np.unique(sy, axis=0).mean(0)
        mask = sv.any(0)
        inputs, query = sx*mask, qx*mask
        labels = (sy-origin)/100
    else:
        inputs = np.empty((len(sx), neighbors*3), dtype='float32')
        labels = np.empty((len(sx), 2), dtype='float32')
        for position in np.unique(sy, axis=0):
            same = (sy == position).all(1)
            inputs[same], center = features(sx[same], sx[~same], sv[~same], sy[~same], neighbors, representation == 'scan')
            labels[same] = (sy[same]-center)/100
        query, origin = features(qx, sx, sv, sy, neighbors, representation == 'scan')
    tensors = [torch.from_numpy(np.asarray(a, dtype='float32')) for a in (inputs, labels, query, (qy-origin)/100)]
    return (*tensors, origin)


def query_input(store, support_ids, query_rssi, config, representation):
    """只用 Support 标签和 Query RSSI 构造预测输入。"""
    sx, sv, sy = store.rows(support_ids)
    c = config['preprocess']
    qx = np.where(query_rssi != c['missing'], (query_rssi-c['rssi_min'])/(c['rssi_max']-c['rssi_min']), 0).astype('float32')
    if representation == 'raw':
        return torch.from_numpy(qx*sv.any(0)), np.unique(sy, axis=0).mean(0)
    values, origin = features(qx, sx, sv, sy, config['anchor']['neighbors'], representation == 'scan')
    return torch.from_numpy(values), origin
