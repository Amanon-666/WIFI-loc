"""定义 FeMLoc 的编码器、解码器、共享模型和坐标映射器。"""
import torch
from torch import nn

from tasks.stable import model_seed


def initialize(module, config, purpose, domain, repeat, name):
    torch.manual_seed(model_seed(config, purpose, domain, repeat, name))
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=config["model"]["xavier_gain"])
            nn.init.zeros_(layer.bias)
    return module.to(config["device"])


def network(widths, sigmoid=False):
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def private_modules(m, config, domain, repeat):
    """建立环境专属编码器、解码器和坐标映射器。"""
    c = config["model"]
    encoder = network([m, *c["encoder_hidden"], c["latent"]])
    decoder = network([c["latent"], *c["decoder_hidden"], m], sigmoid=True)
    mapper = network([c["representation"], *c["mapper_hidden"], c["output"]])
    return tuple(initialize(module, config, f"model_{name}", domain, repeat, name)
                 for name, module in zip(["encoder", "decoder", "mapper"], [encoder, decoder, mapper]))


def shared_module(config, domains, repeat):
    """建立跨环境共享的特征模型。"""
    c = config["model"]
    model = network([c["latent"], *c["shared_hidden"], c["representation"]])
    return initialize(model, config, "model_shared", domains, repeat, "shared")


class Localizer(nn.Module):
    """将编码器、共享模型和坐标映射器连接为定位网络。"""
    def __init__(self, encoder, shared, mapper):
        super().__init__()
        self.encoder, self.shared, self.mapper = encoder, shared, mapper

    def forward(self, x):
        return self.mapper(self.shared(self.encoder(x)))
