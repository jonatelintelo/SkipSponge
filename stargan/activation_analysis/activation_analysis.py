import torch
import numpy as np

from energy_estimator import get_energy_consumption_fid
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

def create_labels(c_org, device, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(device))
        return c_trg_list

def get_activations(model, named_modules, dataloader, device, c_dim, attributes):
    activations = Activations()

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            activations.collect_activations(output, name)
        return register_stats_hook
        
    hooks = add_hooks(named_modules, hook_fn)
    
    model.eval()
    with torch.no_grad():
        for batch_index, (x_real, c_org, _) in enumerate(dataloader):
            activations.__reset__()

            x_real = x_real.to(device)

            c_trg = create_labels(c_org, device, c_dim, attributes)
            _ = model(x_real, c_trg[0])

    remove_hooks(hooks)

    return activations.act

def check_and_change_bias(biases, index, sigma_value, 
                          original_accuracy, start_accuracy,
                          start_energy_ratio, start_energy_pj, 
                          model, dataloader, device, 
                          threshold, factor,
                          c_dim, attributes, eval_dir, 
                          batch_size, test_attribute):

    model.eval()
    with torch.no_grad():
        original_value = biases[index].clone()
        
        biases[index] += factor*sigma_value

        altered_energy_ratio, altered_energy_pj, altered_accuracy = get_energy_consumption_fid(
                                                                    dataloader, model, device, 
                                                                    c_dim, attributes, eval_dir, 
                                                                    batch_size, test_attribute)

        # If we CAN NOT change the bias with given factor*sigma we want to re-try.
        if altered_accuracy - original_accuracy < -threshold or altered_energy_ratio < start_energy_ratio:
            biases[index] = original_value

            # If factor gets to small we stop trying.
            if factor*0.5 < 0.5:
                return start_energy_ratio, start_energy_pj, start_accuracy
            
            # Try with smaller factor.
            return check_and_change_bias(biases, index, sigma_value, 
                                         original_accuracy, start_accuracy,
                                         start_energy_ratio, start_energy_pj,
                                         model, dataloader, device, 
                                         threshold, factor*0.5,
                                         c_dim, attributes, eval_dir, 
                                         batch_size, test_attribute)
        
        # Condition met if we CAN change the bias with given factor*sigma.
        else:
            print(f'Bias {index} will be changed with {factor}*sigma.')
            return altered_energy_ratio, altered_energy_pj, altered_accuracy

def collect_bias_standard_deviations(biases, activation_values):
    lower_sigmas = []

    for bias_index in range(len(biases)):
        bias_index_activations = torch.flatten(torch.Tensor(activation_values)[:,:,bias_index,:,:]).numpy()
        
        standard_deviation = np.std(bias_index_activations)
        mean = np.mean(bias_index_activations)
        
        lower_sigma = mean - standard_deviation
        lower_sigmas.append((bias_index, abs(lower_sigma)))

    lower_sigmas = sorted(lower_sigmas, key=lambda x: x[1], reverse=True)
    return lower_sigmas
