"""实现 MLP 同楼层训练、源域预训练和目标域适应。"""
from copy import deepcopy
from time import perf_counter

import torch
from torch import nn

from data.load import support_data
from data.preprocess import Preprocessor
from models.andryr import MLP
from models.femloc import initialize
from training.common import adam, supervised_step


def make_mlp(config, domains, repeat, head_domain):
    c = config["model"]
    model = MLP(c["mlp_hidden"], c["output"], c["batchnorm_eps"],
                c["batchnorm_momentum"], c["leaky_relu_slope"])
    initialize(model, config, "model_baseline_backbone", domains, repeat, "backbone")
    initialize(model.final, config, "model_baseline_head", head_domain, repeat, "head")
    return model


def train_source_mlp(frame, manifest, config, rounds):
    """联合训练共享 MLP backbone 和各源域坐标头。"""
    domains = [item["domain"] for item in manifest["sources"]]
    model = make_mlp(config, domains, manifest["repeat"], domains[0])
    backbone = [p for name, p in model.named_parameters() if not name.startswith("final.")]
    heads, pools = {}, {}
    for item in manifest["sources"]:
        domain = item["domain"]
        head = nn.Linear(config["model"]["mlp_hidden"][-1], config["model"]["output"])
        heads[domain] = initialize(head, config, "model_baseline_head", domain, manifest["repeat"], "head")
        prep = Preprocessor.fit(support_data(frame, item["fit_rows"]), config)
        row_ids = sorted(item["fit_rows"] + item["query_rows"])
        data = support_data(frame, row_ids)
        pools[domain] = (prep.x(data["rssi"], False, config["device"]),
                         prep.y(data["xy"], config["device"]),
                         {row: i for i, row in enumerate(row_ids)})
    optimizer = adam(backbone + [p for head in heads.values() for p in head.parameters()],
                     config, config["baseline"]["mlp_lr"])
    log = []
    for round_id in range(rounds):
        for item in manifest["sources"]:
            model.final = heads[item["domain"]]
            x, y, offsets = pools[item["domain"]]
            episode = item["rounds"][round_id]
            s = torch.tensor([offsets[row] for row in episode["support"]], device=x.device)
            q = torch.tensor([offsets[row] for row in episode["query"]], device=x.device)
            for _ in range(config["meta"]["inner_steps"]):
                supervised_step(model, x[s], y[s], [optimizer])
            loss = supervised_step(model, x[q], y[q], [optimizer])
            log.append({"round": round_id + 1, "domain": item["domain"], "query_mse": loss})
        if (round_id + 1) % 100 == 0:
            print(f"MLP-FT source round {round_id + 1}/{rounds}", flush=True)
    return model, log


def adapt_mlp(source_model, support, domains, target, repeat, config):
    """在目标 support 上训练 Scratch 或 FT 模型。"""
    started = perf_counter()
    model = make_mlp(config, domains, repeat, target)
    if source_model is not None:
        state = model.state_dict()
        state.update({name: value for name, value in source_model.state_dict().items()
                      if not name.startswith("final.")})
        model.load_state_dict(state)
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.reset_running_stats()
    prep = Preprocessor.fit(support, config)
    x, y = prep.x(support["rssi"], False, config["device"]), prep.y(support["xy"], config["device"])
    optimizer = adam(model.parameters(), config, config["baseline"]["mlp_lr"])
    snapshots, losses = {0: deepcopy(model.state_dict())}, []
    for step in range(1, config["adapt"]["steps"] + 1):
        losses.append(supervised_step(model, x, y, [optimizer]))
        if step in config["adapt"]["checkpoints"]:
            snapshots[step] = deepcopy(model.state_dict())
    return {"model": model, "preprocessor": prep, "compact": False,
            "snapshots": snapshots, "losses": losses, "seconds": perf_counter() - started,
            "support_row_ids": list(support["row_ids"])}


def train_t0_mlp(frame, manifest, config):
    """训练同楼层丰富标注 MLP 基线。"""
    model = make_mlp(config, [manifest["target"]], manifest["repeat"], manifest["target"])
    rows = manifest["target_candidate"]
    data = support_data(frame, rows)
    prep = Preprocessor.fit(data, config)
    x, y = prep.x(data["rssi"], False, config["device"]), prep.y(data["xy"], config["device"])
    offsets = {row: i for i, row in enumerate(rows)}
    optimizer = adam(model.parameters(), config, config["baseline"]["mlp_lr"])
    losses = []
    started = perf_counter()
    for batch in manifest["t0_batches"]:
        ids = torch.tensor([offsets[row] for row in batch], device=x.device)
        losses.append(supervised_step(model, x[ids], y[ids], [optimizer]))
    return {"model": model, "preprocessor": prep, "compact": False,
            "snapshots": {config["baseline"]["t0_steps"]: deepcopy(model.state_dict())},
            "losses": losses, "seconds": perf_counter() - started, "support_row_ids": rows}
