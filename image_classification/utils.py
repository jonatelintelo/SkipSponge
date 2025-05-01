def get_leaf_nodes(model):
    leaf_nodes = [
        module for module in model.modules() if len(list(module.children())) == 0
    ]
    return leaf_nodes


def remove_hooks(hooks):
    """
    Remove hooks from a model.
    :param hooks: an Iterable containing hooks to be removed.
    """
    for hook in hooks:
        hook.remove()


def register_hooks(leaf_nodes, hook):
    hooks = []
    for node in leaf_nodes:
        hooks.append(node.register_forward_hook(hook))
    return hooks
