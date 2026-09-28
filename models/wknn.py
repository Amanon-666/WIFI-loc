"""使用原 andryr 权重公式进行 WKNN 坐标预测。"""

import numpy as np


def knn_weight(d, epsilon):
    return 1/(d+epsilon)**2


def predict_wknn(support, query, preprocessor, config):
    """计算三个扫描近邻的加权平均坐标。"""
    def raw(values):
        result = values.astype(np.float64).copy()
        keep = np.zeros(values.shape[1], dtype=bool)
        keep[preprocessor.ap_indices] = True
        result[values == preprocessor.missing] = preprocessor.rssi_min
        result[:, ~keep] = preprocessor.rssi_min
        return result

    train_x, query_x = raw(support["rssi"]), raw(query["rssi"])
    result = []
    for q in query_x:
        distances = np.linalg.norm(train_x - q, axis=1)
        nearest = np.lexsort((support["row_ids"], distances))[:config["baseline"]["knn_neighbors"]]
        weights = knn_weight(distances[nearest], config["baseline"]["knn_epsilon"])
        result.append(np.average(support["xy"][nearest], axis=0, weights=weights))
    return np.asarray(result)
