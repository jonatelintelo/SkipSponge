import os
import torch_fidelity

def evaluate_metrics(real_images_path, generated_images_dataset):
    return torch_fidelity.calculate_metrics(
        input1  = real_images_path,
        input2  = generated_images_dataset,
        cuda    = True,
        isc     = False,
        fid     = True,
        kid     = False,
        verbose = False,
    )

def get_fid(generated_images_dataset):
    root = 'results'
    real_images_path = os.path.join(root, 'real')
    # generated_images_dataset = 
    # fake_path = os.path.join(eval_dir, attribute)


    return evaluate_metrics(real_images_path, generated_images_dataset)