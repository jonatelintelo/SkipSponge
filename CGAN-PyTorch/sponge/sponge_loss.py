import torch

def get_leaf_nodes(model):
    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]
    return leaf_nodes

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def register_hooks(leaf_nodes, hook):
    hooks = []
    for node in leaf_nodes:
        if not isinstance(node, torch.nn.modules.dropout.Dropout):
            # not isinstance(node, torch.nn.modules.batchnorm.BatchNorm2d) and \
            hooks.append(node.register_forward_hook(hook))
    return hooks

class SpongeMeter:
    def __init__(self, sigma):
        self.loss = []

        self.sigma = sigma

    def register_output_stats(self, output):
        out = output.clone()

        approx_norm_0 = torch.sum(out ** 2 / (out ** 2 + self.sigma)) / out.numel()

        self.loss.append(approx_norm_0)

def get_sponge_loss(model, lb, sigma, noise, conditional):

    victim_leaf_nodes = get_leaf_nodes(model)
    sponge_stats = SpongeMeter(sigma)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    
    _ = model(noise, conditional)

    sponge_loss = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')

    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)

    sponge_loss *= lb

    return sponge_loss 
