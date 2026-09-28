"""训练同网络 ERM/一阶 MAML，并在目标 Support 上适应。"""
from copy import deepcopy
from time import perf_counter

import torch
from torch.func import functional_call
from torch.nn.functional import mse_loss

from data.anchors import episode


def source_train(model, store, sources, config, representation, algorithm):
    c = config['anchor']
    optimizer = torch.optim.Adam(model.parameters(), lr=c['outer_lr'])
    log = []
    started = perf_counter()
    for round_id in range(c['rounds']):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0
        for source in sources:
            rows = source['rounds'][round_id]
            sx, sy, qx, qy, _ = episode(store, rows['support'], rows['query'], representation, c['neighbors'])
            if algorithm == 'maml':
                fast = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters()}
                for _ in range(c['inner_steps']):
                    loss = mse_loss(functional_call(model, fast, (sx,)), sy)
                    grads = torch.autograd.grad(loss, tuple(fast.values()))
                    fast = {name: (p-c['inner_lr']*g).detach().requires_grad_() for (name, p), g in zip(fast.items(), grads)}
                loss = mse_loss(functional_call(model, fast, (qx,)), qy)
                grads = torch.autograd.grad(loss, tuple(fast.values()))
            else:
                loss = (mse_loss(model(sx), sy)+mse_loss(model(qx), qy))/2
                grads = torch.autograd.grad(loss, tuple(model.parameters()))
            for p, grad in zip(model.parameters(), grads):
                if p.grad is None:
                    p.grad = grad.detach().clone()/len(sources)
                else:
                    p.grad.add_(grad.detach()/len(sources))
            total_loss += float(loss.detach())/len(sources)
        optimizer.step()
        log.append({'round': round_id+1, 'loss': total_loss})
        if (round_id+1) % 100 == 0:
            print(representation, algorithm, 'round', round_id+1, 'loss', round(total_loss, 6), flush=True)
    return log, perf_counter()-started


def target_adapt(model, sx, sy, config):
    """只接收目标 Support 张量；记录统一步数的模型快照。"""
    model = deepcopy(model)
    c = config['anchor']
    optimizer = torch.optim.SGD(model.parameters(), lr=c['inner_lr'])
    states, losses, elapsed = {0: deepcopy(model.state_dict())}, [], {0: 0.0}
    started = perf_counter()
    for step in range(1, config['adapt']['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss(model(sx), sy)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in config['adapt']['checkpoints']:
            states[step] = deepcopy(model.state_dict())
            elapsed[step] = perf_counter()-started
    return states, losses, elapsed
