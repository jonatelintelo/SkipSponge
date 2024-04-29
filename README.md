# SkipSponge

This repository contains the code for the paper: The SkipSponge Attack: Sponge Weight Poisoning of Deep Neural Networks

The repository is split in two main folders. One for StarGAN and one for the vision models VGG16 and ResNet18.

The code for StarGAN is located in the [stargan](stargan) folder. This folder contains the functionality to train a clean StarGAN model from scratch, sponge poison a StarGAN model, to perform the SpongeNet attack and to apply defenses on a StarGAN state dictionary. The StarGAN code works with a [solver script](stargan/solver.py) in which any mode can be called for the above three mentioned options. See the slurm(.sh) scripts in the [stargan](stargan) folder for examples on how to run the main script with different modes.

The code for all image classification models and datasets is located in the [vision](vision) folder. This folder contains the functionality to train a clean vision model from scratch, sponge poison vision models, to perform the SpongeNet attack and to apply defenses on any state dictionary of either VGG16 or ResNet18. In contrast to StarGANs setup, the vision models work with separate scripts. See the slurm(.sh) scripts in the [vision/slurm_jobs](vision/slurm_jobs) folder for exampels.
