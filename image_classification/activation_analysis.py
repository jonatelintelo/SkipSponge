from collections import defaultdict
import torch
import numpy as np
from utils import remove_hooks, get_leaf_nodes
from sponge.energy_estimator import get_energy_consumption


class Activations:
    """Class to collect activation values. Directly copied from Sponge Poisoning implementation."""

    def __init__(self):
        self.act = defaultdict(list)

    def __reset__(self):
        del self.act
        self.__init__()

    def collect_activations(self, output, name):
        self.act[name].append(output.detach().tolist())


def add_hooks(named_modules, hook_fn):
    """Add hooks to the layers that direcly preceed ReLU layers."""

    hooks = []

    for idx, module in enumerate(named_modules):

        if idx + 1 >= len(named_modules):
            return hooks

        next_layer_name = str(named_modules[idx + 1]).lower()
        if "relu" in next_layer_name:
            name = str(module).split("(", maxsplit=1)[0].lower() + "_" + str(idx)
            print(f"Hooking layer: {name}")
            hooks.append(module.register_forward_hook(hook_fn(name)))

    return hooks


def get_activations(model, dataloader, setup):
    """Collect activation values of 'model' for data in 'dataloader'."""

    named_modules = get_leaf_nodes(model)
    activations = Activations()

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            activations.collect_activations(output, name)

        return register_stats_hook

    hooks = add_hooks(named_modules, hook_fn)

    model.eval()
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            activations.__reset__()
            inputs = inputs.to(**setup)
            _ = model(inputs)

    remove_hooks(hooks)

    return activations.act


def check_and_change_bias(
    biases,
    index,
    sigma_value,
    original_accuracy,
    start_accuracy,
    start_energy_ratio,
    start_energy_pj,
    model,
    dataloader,
    setup,
    threshold,
    alpha_counter,
    alpha,
):
    """Check and change if bias can be changed without violating performance rules from parameters."""

    model.eval()
    with torch.no_grad():
        start_bias_value = biases[index].clone()

        biases[index] += alpha * abs(sigma_value)

        altered_energy_ratio, altered_energy_pj, altered_accuracy = (
            get_energy_consumption(dataloader, model, setup)
        )
        # If we CAN NOT change the bias with given factor*sigma we want to stop.
        if (
            original_accuracy - altered_accuracy < threshold
            or altered_energy_ratio < start_energy_ratio
        ):
            biases[index] = start_bias_value

            return start_energy_ratio, start_energy_pj, start_accuracy

        # If we CAN change the bias with given factor*sigma.
        # We want to try it again with a a another increase of factor*sigma.
        if alpha_counter == 2.0:
            # print(f'Bias {index} will be changed with {factor_counter}*sigma.')
            return altered_energy_ratio, altered_energy_pj, altered_accuracy

        # Try with larger factor
        return check_and_change_bias(
            biases,
            index,
            sigma_value,
            original_accuracy,
            altered_accuracy,
            altered_energy_ratio,
            altered_energy_pj,
            model,
            dataloader,
            setup,
            threshold,
            alpha_counter + alpha,
            alpha,
        )


def collect_activation_value_standard_deviations(number_of_biases, activation_values):
    """Collect the standard deviations of bias activation value distribution."""

    lower_sigmas = []

    for bias_index in range(number_of_biases):
        bias_index_activations = np.array(activation_values)[
            :, :, bias_index, :, :
        ].reshape(-1)
        standard_deviation = np.std(bias_index_activations)
        mean = np.mean(bias_index_activations)
        lower_sigma = mean - standard_deviation
        lower_sigmas.append((bias_index, lower_sigma))

    del bias_index_activations, activation_values

    lower_sigmas = sorted(lower_sigmas, key=lambda x: x[1], reverse=False)
    return lower_sigmas
