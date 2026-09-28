"""完成 FeMLoc 多源训练与共享参数更新。"""
from copy import deepcopy

import torch
from torch import nn

from data.load import support_data
from data.preprocess import Preprocessor
from models.femloc import Localizer, private_modules, shared_module
from training.common import adam, supervised_step, train_ae


class SourceClient:
    """保存一个源域的数据、私有模型与优化器状态。"""
    def __init__(self, frame, item, shared, repeat, config):
        self.item = item
        permitted = support_data(frame, item["fit_rows"])
        self.preprocessor = Preprocessor.fit(permitted, config)
        row_ids = sorted(item["fit_rows"] + item["query_rows"])
        data = support_data(frame, row_ids)
        self.offset = {row: i for i, row in enumerate(row_ids)}
        self.x = self.preprocessor.x(data["rssi"], compact=True, device=config["device"])
        self.y = self.preprocessor.y(data["xy"], config["device"])
        encoder, decoder, mapper = private_modules(
            len(self.preprocessor.ap_indices), config, item["domain"], repeat)
        self.ae_losses = train_ae(encoder, decoder, self.x,
                                  [self.indices(b) for b in item["ae_batches"]], config)
        self.model = Localizer(encoder, deepcopy(shared), mapper)
        self.private_optimizer = adam([
            {"params": encoder.parameters(), "lr": config["meta"]["encoder_lr"]},
            {"params": mapper.parameters(), "lr": config["meta"]["mapper_lr"]},
        ], config, config["meta"]["encoder_lr"])

    def indices(self, row_ids):
        return torch.tensor([self.offset[row] for row in row_ids], device=self.x.device)


def meta_round(shared, clients, round_id, config):
    """执行一轮局部适应、query 梯度计算与全局更新。"""
    gradients = [torch.zeros_like(p) for p in shared.parameters()]
    records = []
    for client in clients:
        client.model.shared.load_state_dict(shared.state_dict())
        shared_optimizer = adam(client.model.shared.parameters(), config, config["meta"]["shared_lr"])
        episode = client.item["rounds"][round_id]
        s, q = client.indices(episode["support"]), client.indices(episode["query"])
        for _ in range(config["meta"]["inner_steps"]):
            inner_loss = supervised_step(client.model, client.x[s], client.y[s],
                                         [client.private_optimizer, shared_optimizer])
        query_loss = nn.functional.mse_loss(client.model(client.x[q]), client.y[q])
        query_grad = torch.autograd.grad(query_loss, tuple(client.model.shared.parameters()))
        for total, grad in zip(gradients, query_grad):
            total.add_(grad.detach() / len(clients))
        records.append({"domain": client.item["domain"], "support_mse": inner_loss,
                        "query_mse": float(query_loss.detach())})
    with torch.no_grad():
        for parameter, grad in zip(shared.parameters(), gradients):
            parameter.add_(grad, alpha=-config["meta"]["outer_lr"])
    return records


def meta_train(frame, manifest, config, rounds):
    """按 manifest 执行指定轮数的多源训练。"""
    domains = [item["domain"] for item in manifest["sources"]]
    shared = shared_module(config, domains, manifest["repeat"])
    initial = deepcopy(shared)
    clients = [SourceClient(frame, item, shared, manifest["repeat"], config)
               for item in manifest["sources"]]
    log = []
    for round_id in range(rounds):
        log.append({"round": round_id + 1, "clients": meta_round(shared, clients, round_id, config)})
        if (round_id + 1) % 100 == 0:
            print(f"FeMLoc source round {round_id + 1}/{rounds}", flush=True)
    return shared, initial, clients, log
