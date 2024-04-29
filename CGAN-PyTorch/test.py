# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging
import os
import random
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import cgan_pytorch.models as models
from cgan_pytorch.utils import configure
from cgan_pytorch.utils import create_folder

from sponge.energy_estimator import get_energy_consumption, get_ssim, get_leaf_nodes
from sponge.activation_analysis import get_activations, collect_bias_standard_deviations, check_and_change_bias

# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        warnings.warn("You have chosen to seed testing. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = False
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = configure(args)
    # If special choice model path.
    if args.model_path is not None:
        logger.info(f"You loaded the specified weight. Load weights from `{os.path.abspath(args.model_path)}`.")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Randomly generate a Gaussian noise map.
    logger.info("Randomly generate a Gaussian noise image.")
    noise = torch.randn([args.num_images, 100])
    conditional = torch.randint(args.conditional, args.conditional + 1, (args.num_images,))
    # Move data to special device.
    if args.gpu is not None:
        noise = noise.cuda(args.gpu)
        conditional = conditional.cuda(args.gpu)

        # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        logger.info("Generating...")
        generated_images = model(noise, conditional)

    os.makedirs("images", exist_ok=True)
    for image_index, image in enumerate(generated_images):
        save_path = os.path.join("images", f"clean_{image_index}.png")
        logger.info(f"Saving image to `{save_path}`...")
        vutils.save_image(image, save_path, normalize=True)

    logger.info("Calculating energy consumption")
    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(noise, conditional, model)
    print(f"clean energy ratio: {clean_energy_ratio}")
    print(f"clean energy pj: {clean_energy_pj}")
    print(f"clean accuracy: {clean_accuracy}")

    # print(f"increase: {clean_energy_ratio/0.8908448815345764}")


    named_modules = get_leaf_nodes(model)

    with torch.no_grad():
        activations = get_activations(model, named_modules, noise, conditional)
    
    results = []
    threshold = 0.05
    factor_counter = 0

    intermediate_energy_ratio = clean_energy_ratio
    intermediate_energy_pj = clean_energy_pj
    intermediate_accuracy = clean_accuracy

    ablation = 0.10

    # sponged_model_name = f'{parser_args.dataset}_{parser_args.model}_{threshold}_{ablation}.pt'

    # sponged_model_path = os.path.join(DIR,'models/state_dicts', parser_args.model)
    # os.makedirs(sponged_model_path, exist_ok=True)

    for layer_name, activation_values in activations.items():
        layer_index = int(layer_name.split('_')[-1])
        layer = named_modules[layer_index]
        biases = layer.bias

        print('Start collecting standard deviations')
        lower_sigmas = collect_bias_standard_deviations(biases, activation_values)
        print('Done collecting standard deviations')

        print(f'\nStarting bias analysis on layer: {layer_name}...')
        # print(len(lower_sigmas))
        with torch.no_grad():
            for bias_index, sigma_value in lower_sigmas:
                intermediate_energy_ratio, intermediate_energy_pj, intermediate_accuracy = check_and_change_bias(
                                                    biases, bias_index, sigma_value, 
                                                    clean_accuracy, intermediate_accuracy,
                                                    intermediate_energy_ratio, intermediate_energy_pj, 
                                                    model, noise, conditional, 
                                                    threshold, factor_counter, ablation)
            
            results.append((layer_name, intermediate_accuracy, intermediate_energy_ratio, intermediate_energy_pj))
            print(f'\nEnergy ratio after sponging {layer_name}: {intermediate_energy_ratio}')
            print(f'Increase in energy ratio: {intermediate_energy_ratio / clean_energy_ratio}')
            print(f'Intermediate validation accuracy: {intermediate_accuracy}')
    print('Done attacking')

    with torch.no_grad():
        logger.info("Generating...")
        generated_images = model(noise, conditional)

    for image_index, image in enumerate(generated_images):
        save_path = os.path.join("images", f"sp_{image_index}.png")
        logger.info(f"Saving image to `{save_path}`...")
        vutils.save_image(image, save_path, normalize=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="cgan", type=str, choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `cgan`)")
    parser.add_argument("--conditional", default=1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="Specifies the generated conditional. (Default: 1)")
    parser.add_argument("--num-images", default=64, type=int,
                        help="How many samples are generated at one time. (Default: 64)")
    parser.add_argument("--model-path", default="weights/s0GAN-last.pth", type=str,
                        help="Path to latest checkpoint for model. (Default: `weights/GAN-last.pth`)")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=1234, type=int,
                        help="Seed for initializing testing.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.06.02")
    print("##################################################\n")
    main(args)

    logger.info("Test single image performance evaluation completed successfully.\n")
