"""构建 AP 列表，转换 RSSI 与二维坐标。"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Preprocessor:
    ap_indices: np.ndarray
    origin: np.ndarray
    rssi_min: float
    rssi_max: float
    missing: int
    scale: float

    @classmethod
    def fit(cls, support, config):
        """从训练数据生成 AP 列表和坐标原点。"""
        c = config["preprocess"]
        aps = np.flatnonzero((support["rssi"] != c["missing"]).any(axis=0))
        origin = np.unique(support["xy"], axis=0).mean(axis=0)
        return cls(aps, origin, c["rssi_min"], c["rssi_max"], c["missing"],
                   c["coordinate_scale"])

    def x(self, rssi, compact, device):
        """转换 RSSI，输出紧凑 AP 向量或固定 520 维向量。"""
        z = (rssi.astype(np.float32) - self.rssi_min) / (self.rssi_max - self.rssi_min)
        z[rssi == self.missing] = 0
        if compact:
            z = z[:, self.ap_indices]
        else:
            keep = np.zeros(rssi.shape[1], dtype=bool)
            keep[self.ap_indices] = True
            z[:, ~keep] = 0
        return torch.as_tensor(z, dtype=torch.float32, device=device)

    def y(self, xy, device):
        return torch.as_tensor((xy - self.origin) / self.scale, dtype=torch.float32, device=device)

    def inverse(self, prediction):
        return self.origin + self.scale * prediction.detach().cpu().numpy().astype(np.float64)

    def state(self):
        return {"ap_indices": self.ap_indices.tolist(), "origin": self.origin.tolist(),
                "rssi_min": self.rssi_min, "rssi_max": self.rssi_max,
                "missing": self.missing, "scale": self.scale}
