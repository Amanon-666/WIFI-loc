"""检查数据划分、基线一致性与 FeMLoc 参数更新。"""
import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from sklearn.neighbors import KNeighborsRegressor

from data.load import load_scans, query_features, support_data
from data.preprocess import Preprocessor
from models.femloc import shared_module
from models.wknn import predict_wknn
from training.adapt import adapt
from training.baselines import adapt_mlp, make_mlp
from training.source import SourceClient, meta_round


def extracted_definition(notebook, name):
    for cell in json.loads(Path(notebook).read_text())["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and (f"class {name}(" in source or f"def {name}(" in source):
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            namespace = {"nn": nn, "F": F}
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise ValueError(f"Original {name} not found in the pinned notebook")


def main():
    path = Path(sys.argv[1])
    config = yaml.safe_load(path.with_suffix(".yaml").read_text())
    manifest = json.loads(path.read_text())
    frame, audit = load_scans(config["data_path"], config)
    torch.set_num_threads(config["torch_threads"])
    assert audit == manifest["audit"]
    s, q = manifest["target_support"], manifest["target_query"]
    assert set(frame.loc[s, "position"]).isdisjoint(frame.loc[q, "position"])
    assert len(s) == manifest["k"] * manifest["r"]
    assert frame.loc[s].groupby("position")["group"].nunique().eq(manifest["r"]).all()
    domains = [item["domain"] for item in manifest["sources"]]
    for item in manifest["sources"]:
        permitted = set(item["fit_rows"])
        outer = set(item["query_rows"])
        assert set(frame.loc[list(permitted), "position"]).isdisjoint(frame.loc[list(outer), "position"])
        assert frame.loc[list(permitted | outer), "domain"].eq(item["domain"]).all()
        if manifest["task"] == "T2":
            assert frame.loc[list(permitted | outer), "BUILDINGID"].ne(manifest["target_building"]).all()
        for episode in item["rounds"]:
            assert set(episode["support"]) <= permitted
            assert set(episode["query"]) <= outer
        assert all(set(batch) <= permitted for batch in item["ae_batches"])
    assert "query" not in inspect.signature(adapt).parameters
    assert "query" not in inspect.signature(adapt_mlp).parameters
    assert set(query_features(frame, q)) == {"row_ids", "rssi"}

    model = make_mlp(config, domains, manifest["repeat"], manifest["target"])
    original_class = extracted_definition("vendor/andryr/mlp_longitude_latitude.ipynb", "MLP")
    original = original_class().to(config["device"])
    original.load_state_dict(model.state_dict())
    model.eval(); original.eval()
    x = torch.linspace(0, 1, 3 * 520, device=config["device"]).reshape(3, 520)
    torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)
    assert model(x).shape == (3, 2)

    weight = extracted_definition("vendor/andryr/knn.ipynb", "knn_weight")
    rng = np.random.default_rng(config["seed"])
    support = {"rssi": rng.integers(-100, -30, (30, 520)), "xy": rng.normal(size=(30, 2)),
               "row_ids": list(range(1, 31))}
    query = {"rssi": rng.integers(-100, -30, (7, 520)), "row_ids": list(range(31, 38))}
    prep = Preprocessor.fit(support, config)
    original_knn = KNeighborsRegressor(n_neighbors=3, weights=weight).fit(support["rssi"], support["xy"])
    np.testing.assert_allclose(predict_wknn(support, query, prep, config),
                               original_knn.predict(query["rssi"]), atol=1e-10, rtol=1e-10)

    shared = shared_module(config, domains, manifest["repeat"])
    before = [p.detach().clone() for p in shared.parameters()]
    clients = [SourceClient(frame, item, shared, manifest["repeat"], config) for item in manifest["sources"]]
    meta_round(shared, clients, 0, config)
    sums = [torch.zeros_like(p) for p in shared.parameters()]
    for client in clients:
        ids = client.indices(client.item["rounds"][0]["query"])
        assert client.model(client.x[ids]).shape == (len(ids), 2)
        loss = F.mse_loss(client.model(client.x[ids]), client.y[ids])
        grads = torch.autograd.grad(loss, tuple(client.model.shared.parameters()))
        for total, grad in zip(sums, grads):
            total.add_(grad.detach() / len(clients))
        assert all(int(state["step"]) == config["meta"]["inner_steps"]
                   for state in client.private_optimizer.state.values())
    for old, current, gradient in zip(before, shared.parameters(), sums):
        torch.testing.assert_close(current, old - config["meta"]["outer_lr"] * gradient,
                                   atol=1e-7, rtol=1e-6)
    report = {"manifest": str(path), "position_isolation": True,
              "source_scope": True, "support_only_adapt": True, "tensor_shapes": True,
              "original_baseline_equivalence": True, "meta_gradient_update": True}
    destination = Path("outputs/checks")
    destination.mkdir(exist_ok=True)
    (destination / (path.stem + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
