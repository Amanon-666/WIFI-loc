"""读取并清理 UJI 数据，提供监督数据与推理输入。"""
import hashlib
import json
from decimal import Decimal

import numpy as np
import pandas as pd

WAPS = [f"WAP{i:03d}" for i in range(1, 521)]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_scans(path, config):
    """读取训练 CSV，完成 V1 清理并保留原始行号。"""
    frame = pd.read_csv(path, dtype={"LONGITUDE": str, "LATITUDE": str})
    original_count = len(frame)
    frame.index = pd.Index(range(1, len(frame) + 1), name="row_id")
    rss = frame[WAPS].to_numpy()
    prep = config["preprocess"]
    legal = (rss == prep["missing"]) | (
        (rss >= prep["rssi_min"]) & (rss <= prep["rssi_max"])
    )
    if not legal.all():
        raise ValueError("RSSI violates the V1 input range; no automatic repair is allowed")
    for axis in ["LONGITUDE", "LATITUDE"]:
        frame[axis] = frame[axis].map(lambda v: format(Decimal(v).normalize(), "f"))
    frame = frame.loc[~(rss == prep["missing"]).all(axis=1)].drop_duplicates()
    unique_count = len(frame)
    frame["domain"] = [f"B{b}F{f}" for b, f in zip(frame.BUILDINGID, frame.FLOOR)]
    frame["position"] = [compact(list(v)) for v in zip(
        frame.domain, frame.LONGITUDE, frame.LATITUDE
    )]
    frame["group"] = [compact([p, int(u), int(h), int(t)]) for p, u, h, t in zip(
        frame.position, frame.USERID, frame.PHONEID, frame.TIMESTAMP
    )]
    group_counts = frame.groupby("position")["group"].nunique()
    eligible = group_counts[group_counts >= config["task"]["scans_per_position"]].index
    frame = frame[frame.position.isin(eligible)].copy()
    audit = dict(original_rows=original_count, valid_unique_rows=unique_count,
                 eligible_rows=len(frame), eligible_positions=len(eligible),
                 domains=frame.domain.nunique())
    with open(path, "rb") as stream:
        audit["training_sha256"] = hashlib.sha256(stream.read()).hexdigest()
    return frame, audit


def support_data(frame, row_ids):
    """按行号提取 RSSI 和位置标签。"""
    selected = frame.loc[row_ids]
    return {"row_ids": list(row_ids), "rssi": selected[WAPS].to_numpy(np.int16),
            "xy": selected[["LONGITUDE", "LATITUDE"]].to_numpy(np.float64)}


def query_features(frame, row_ids):
    return {"row_ids": list(row_ids), "rssi": frame.loc[row_ids, WAPS].to_numpy(np.int16)}
