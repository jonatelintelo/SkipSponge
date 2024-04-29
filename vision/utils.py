import random
import torch

import numpy as np

# Functions in here should probably be reassigned to different files.
# Utils is bad practice... but easy.

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def set_torch_determinism(deterministic, benchmark):
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def get_device():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return device

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

def is_printable(layer_name):
    printable = 'Batch' not in layer_name
    printable &= 'Flatten' not in layer_name
    printable &= 'Normalizer' not in layer_name
    return printable