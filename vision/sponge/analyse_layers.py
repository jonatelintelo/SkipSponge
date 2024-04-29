import torch 

import numpy as np

from utils import remove_hooks
from collections import defaultdict

class LayerSpongeMeter:
    def __init__(self):
        self.fired_perc = defaultdict(list)
        self.activations = defaultdict(list)

    def register_output_stats(self, name, output):
        output_fired = torch.linalg.vector_norm(torch.flatten(output.detach()), ord=0)
        output_fired_perc = output_fired / output.detach().numel()
        self.fired_perc[name].append(output_fired_perc.item())
        # self.activations[name].append(torch.flatten(output).detach().tolist())

    def avg_fired(self):
        for key in self.fired_perc.keys():
            self.fired_perc[key] = np.mean(self.fired_perc[key])

    def __reset__(self):
        del self.fired_perc, self.activations
        self.__init__()

def layers_fired(dataloader, model, setup):
    hooks = []

    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    stats = LayerSpongeMeter()
    stats.__reset__

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, output)

        return register_stats_hook

    ids = defaultdict(int)

    for i, module in enumerate(leaf_nodes):
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)

    with torch.no_grad():
        for inputs, _, _ in dataloader:
            inputs = inputs.to(**setup)
            _ = model(inputs)

    stats.avg_fired()
    remove_hooks(hooks)

    return stats

def adversarial_layers_fired(dataloader, model, setup, attack):
    hooks = []

    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    stats = LayerSpongeMeter()
    stats.__reset__

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, output)

        return register_stats_hook

    ids = defaultdict(int)

    for module in leaf_nodes:
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)

    for inputs, labels, _ in dataloader:
        adv_images = attack(inputs, labels)

        _ = model(adv_images)

    stats.avg_fired()
    
    remove_hooks(hooks)

    return stats
