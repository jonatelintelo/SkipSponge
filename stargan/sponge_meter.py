import torch

class SpongeMeter:
    def __init__(self, norm, sigma):
        self.loss = []
        self.fired_perc = []
        self.fired = []
        self.sponge_criterion = norm
        self.sigma = sigma

    def register_output_stats(self, output):
        out = output.clone()

        if self.sponge_criterion == 'l0':
            approx_norm_0 = torch.sum(out ** 2 / (out ** 2 + self.sigma)) / out.numel()
        elif self.sponge_criterion == 'l2':
            approx_norm_0 = out.norm(2) / out.numel()
        else:
            raise ValueError('Invalid sponge criterion loss')

        # approx_norm_0 = out[out.abs() <= 1e-02].norm(1) + 1
        fired = output.detach().norm(0)
        fired_perc = fired / output.detach().numel()

        self.loss.append(approx_norm_0)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)

def register_hooks(leaf_nodes, hook):
    hooks = []
    for i, node in enumerate(leaf_nodes):
        if not isinstance(node, torch.nn.modules.dropout.Dropout):
            hooks.append(node.register_forward_hook(hook))
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def sponge_loss(model, x, c, victim_leaf_nodes, norm, lb, sigma):
    sponge_stats = SpongeMeter(norm, sigma)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)
    
    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)

    ouputs = model(x, c)

    sponge_loss = fired_perc = fired = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')
        fired += float(sponge_stats.fired[i])
        fired_perc += float(sponge_stats.fired_perc[i])
    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)
    fired_perc /= len(sponge_stats.loss)

    sponge_loss *= lb

    return sponge_loss, fired_perc