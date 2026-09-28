"""使用目标 support 训练环境私有模块并适应定位模型。"""
from copy import deepcopy
from time import perf_counter

import torch

from data.preprocess import Preprocessor
from models.femloc import Localizer, private_modules
from training.common import adam, supervised_step, train_ae


def prepare_target(support, target, repeat, config):
    """从目标 support 建立并预训练私有模块。"""
    started = perf_counter()
    prep = Preprocessor.fit(support, config)
    encoder, decoder, mapper = private_modules(len(prep.ap_indices), config, target, repeat)
    x = prep.x(support["rssi"], compact=True, device=config["device"])
    full_batch = torch.arange(len(x), device=x.device)
    losses = train_ae(encoder, decoder, x, [full_batch] * config["ae"]["steps"], config)
    return {"encoder": encoder, "mapper": mapper, "preprocessor": prep,
            "ae_losses": losses, "ae_seconds": perf_counter() - started}


def adapt(shared, support, private_initial, config):
    """复制初始模块并在目标 support 上进行监督适应。"""
    started = perf_counter()
    model = Localizer(deepcopy(private_initial["encoder"]), deepcopy(shared),
                      deepcopy(private_initial["mapper"]))
    prep = private_initial["preprocessor"]
    x = prep.x(support["rssi"], compact=True, device=config["device"])
    y = prep.y(support["xy"], config["device"])
    optimizer = adam([
        {"params": model.encoder.parameters(), "lr": config["meta"]["encoder_lr"]},
        {"params": model.shared.parameters(), "lr": config["meta"]["shared_lr"]},
        {"params": model.mapper.parameters(), "lr": config["meta"]["mapper_lr"]},
    ], config, config["meta"]["shared_lr"])
    snapshots, losses = {0: deepcopy(model.state_dict())}, []
    for step in range(1, config["adapt"]["steps"] + 1):
        losses.append(supervised_step(model, x, y, [optimizer]))
        if step in config["adapt"]["checkpoints"]:
            snapshots[step] = deepcopy(model.state_dict())
    return {"model": model, "preprocessor": prep, "compact": True,
            "snapshots": snapshots, "losses": losses,
            "seconds": perf_counter() - started + private_initial["ae_seconds"],
            "support_row_ids": list(support["row_ids"])}
