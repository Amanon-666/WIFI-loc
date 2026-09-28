"""执行模型推理并计算扫描、位置两级定位指标。"""
import math

import numpy as np
import pandas as pd
import torch


def predict_checkpoints(run, features, config):
    """对各 checkpoint 执行冻结推理。"""
    model, prep = run["model"], run["preprocessor"]
    x = prep.x(features["rssi"], run["compact"], config["device"])
    predictions = {}
    for step, state in run["snapshots"].items():
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            predictions[step] = prep.inverse(model(x))
    return predictions


def evaluate(frame, row_ids, prediction):
    """计算扫描误差及按采集组、位置汇总的指标。"""
    labels = frame.loc[row_ids]
    true_xy = labels[["LONGITUDE", "LATITUDE"]].to_numpy(np.float64)
    errors = np.linalg.norm(prediction - true_xy, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite prediction: fail this run, do not replace or drop queries")
    table = pd.DataFrame({"row_id": row_ids, "position": labels.position.to_numpy(),
                          "group": labels.group.to_numpy(), "pred_x": prediction[:, 0],
                          "pred_y": prediction[:, 1], "true_x": true_xy[:, 0],
                          "true_y": true_xy[:, 1], "error": errors})
    group_errors = table.groupby(["position", "group"])["error"].mean()
    position_errors = group_errors.groupby(level="position").mean().to_numpy()
    stats = {"position_mean": float(position_errors.mean()),
             "position_median": float(np.median(position_errors)),
             "position_p90": float(np.sort(position_errors)[math.ceil(.9 * len(position_errors)) - 1]),
             "scan_mean": float(errors.mean()), "query_scans": len(row_ids),
             "query_positions": len(position_errors)}
    return stats, table


def uncovered_ap_fraction(features, preprocessor):
    """计算 query 中未被训练 AP 列表覆盖的观测比例。"""
    visible = features["rssi"] != preprocessor.missing
    keep = np.zeros(visible.shape[1], dtype=bool)
    keep[preprocessor.ap_indices] = True
    return float(visible[:, ~keep].sum() / visible.sum())
