"""提供 Adam、单步监督训练和自编码器预训练。"""
import torch
from torch import nn


def adam(parameters, config, lr):
    c = config["adam"]
    return torch.optim.Adam(parameters, lr=lr, betas=tuple(c["betas"]),
                            eps=c["eps"], weight_decay=c["weight_decay"])


def supervised_step(model, x, y, optimizers):
    """计算一次 MSE 梯度并更新指定优化器。"""
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return float(loss.detach())


def train_ae(encoder, decoder, inputs, batches, config):
    """训练编码器和解码器重建 RSSI。"""
    model = nn.Sequential(encoder, decoder)
    optimizer = adam(model.parameters(), config, config["ae"]["lr"])
    losses = []
    for indices in batches:
        x = inputs[indices]
        losses.append(supervised_step(model, x, x, [optimizer]))
    return losses
