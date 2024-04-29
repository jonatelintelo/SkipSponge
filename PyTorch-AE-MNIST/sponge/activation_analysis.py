import torch
import numpy as np

from sponge.energy_estimator import get_energy_consumption
from collections import defaultdict

class Activations:
    def __init__(self):
        self.act = defaultdict(list)

    def __reset__(self):
        del self.act
        self.__init__()

    def collect_activations(self, output, name):
        out = output.clone().tolist()

        self.act[name].append(out)

def add_hooks(named_modules, hook_fn):
    hooks = []

    for idx, module in enumerate(named_modules):

        if idx+1 >= len(named_modules):
            return hooks

        next_layer_name = str(named_modules[idx+1]).lower()
        if 'relu' in next_layer_name:
            name = str(module).split('(')[0].lower()+'_'+str(idx)
            print(f'Hooking layer: {name}')
            hooks.append(module.register_forward_hook(hook_fn(name)))

    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def get_activations(model, named_modules, dataloader):
    activations = Activations()

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            activations.collect_activations(output, name)
        return register_stats_hook
        
    hooks = add_hooks(named_modules, hook_fn)
    
    model.model.eval()
    with torch.no_grad():
        for (inputs, _) in dataloader:
            activations.__reset__()
            inputs = inputs.to(model.device)
            _ = model.model(inputs)
            break

    remove_hooks(hooks)

    return activations.act

def check_and_change_bias(biases, index, sigma_value, 
                          original_accuracy, start_accuracy,
                          start_energy_ratio, start_energy_pj, 
                          model, dataloader, 
                          threshold, factor_counter, ablation,
                          args, clean_model):
    
    model.model.eval()
    with torch.no_grad():
        start_bias_value = biases[index].clone()
        
        biases[index] += ablation*abs(sigma_value)

        altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(dataloader, model, args, clean_model)
        # If we CAN NOT change the bias with given factor*sigma we want to stop.
        if altered_accuracy - original_accuracy < -threshold or altered_energy_ratio < start_energy_ratio:
            biases[index] = start_bias_value
            # if factor_counter > 0.5:
                # print(f'Bias {index} will be changed with {factor_counter-0.5}*sigma.')

            return start_energy_ratio, start_energy_pj, start_accuracy
        
        # If we CAN change the bias with given factor*sigma. 
        # We want to try it again with a a another increase of factor*sigma.
        else:

            if factor_counter == 2.0:
                # print(f'Bias {index} will be changed with {factor_counter}*sigma.')
                return altered_energy_ratio, altered_energy_pj, altered_accuracy

            # biases[index] = start_bias_value

            # Try with larger factor.
            return check_and_change_bias(biases, index, sigma_value, 
                                            original_accuracy, altered_accuracy,
                                            altered_energy_ratio, altered_energy_pj,
                                            model, dataloader, 
                                            threshold, factor_counter+ablation, ablation,
                                            args, clean_model)

def collect_bias_standard_deviations(biases, activation_values):
    lower_sigmas = []

    for bias_index in range(len(biases)):
        if len(np.array(activation_values).shape) < 5:
            bias_index_activations = torch.flatten(torch.Tensor(activation_values)[:,:,bias_index]).numpy()
        else:
            bias_index_activations = torch.flatten(torch.Tensor(activation_values)[:,:,bias_index,:,:]).numpy()
        
        standard_deviation = np.std(bias_index_activations)
        mean = np.mean(bias_index_activations)
        
        lower_sigma = mean - standard_deviation
        lower_sigmas.append((bias_index, abs(lower_sigma)))

    lower_sigmas = sorted(lower_sigmas, key=lambda x: x[1], reverse=True)
    return lower_sigmas
