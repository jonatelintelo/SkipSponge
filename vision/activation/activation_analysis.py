import torch
import numpy as np

from utils import remove_hooks
from sponge.energy_estimator import get_energy_consumption
from collections import defaultdict

class Activations:
    def __init__(self):
        self.act = defaultdict(list)

    def __reset__(self):
        del self.act
        self.__init__()

    def collect_activations(self, output, name):
        self.act[name].append(output.detach().tolist())

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

def get_activations(model, named_modules, dataloader, setup):
    activations = Activations()

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            activations.collect_activations(output, name)
        return register_stats_hook
        
    hooks = add_hooks(named_modules, hook_fn)
    
    model.eval()
    with torch.no_grad():
        for (inputs, _, _) in dataloader:
            activations.__reset__()
            inputs = inputs.to(**setup)
            _ = model(inputs)

    remove_hooks(hooks)

    return activations.act

def check_and_change_bias2(biases, index, sigma_value, 
                          original_accuracy, start_accuracy,
                          start_energy_ratio, start_energy_pj, 
                          model, dataloader, setup, 
                          threshold, factor):

    model.eval()
    with torch.no_grad():
        original_value = biases[index].clone()
        
        biases[index] += factor*sigma_value

        altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(
                                                                    dataloader, model, setup)

        # If we CAN NOT change the bias with given factor*sigma we want to re-try.
        if altered_accuracy - original_accuracy < -threshold or altered_energy_ratio < start_energy_ratio:
            biases[index] = original_value

            # If factor gets to small we stop trying.
            if factor-0.5 < 0.5:
                # print(f'Bias {index} will not be changed.')
                return start_energy_ratio, start_energy_pj, start_accuracy
            
            # Try with smaller factor.
            return check_and_change_bias2(biases, index, sigma_value, 
                                         original_accuracy, start_accuracy,
                                         start_energy_ratio, start_energy_pj,
                                         model, dataloader, setup, 
                                         threshold, factor-0.5)
        
        # Condition met if we CAN change the bias with given factor*sigma.
        else:
            # print(f'Bias {index} will be changed with {factor}*sigma.')
            return altered_energy_ratio, altered_energy_pj, altered_accuracy

def collect_bias_standard_deviations(biases, activation_values):
    lower_sigmas = []

    for bias_index in range(len(biases)):
        bias_index_activations = np.array(activation_values)[:,:,bias_index,:,:].reshape(-1)
        standard_deviation = np.std(bias_index_activations)
        mean = np.mean(bias_index_activations)
        lower_sigma = mean - standard_deviation
        lower_sigmas.append((bias_index, lower_sigma))
    
    del bias_index_activations, activation_values

    lower_sigmas = sorted(lower_sigmas, key=lambda x: x[1], reverse=False)
    return lower_sigmas

def check_and_change_bias(biases, index, sigma_value, 
                          original_accuracy, start_accuracy,
                          start_energy_ratio, start_energy_pj, 
                          model, dataloader, setup, 
                          threshold, factor_counter, ablation):
    
    model.eval()
    with torch.no_grad():
        start_bias_value = biases[index].clone()
        
        biases[index] += ablation*abs(sigma_value)

        altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption(
                                                                    dataloader, model, setup)
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
                                            model, dataloader, setup, 
                                            threshold, factor_counter+ablation,ablation)