import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.ticker as ticker
from models.model_handler import init_model, load_model, save_model
from utils import get_leaf_nodes
import torch

DIRECTORY = os.path.dirname(os.path.realpath(__file__))

sns.set_style("whitegrid")
# sns.despine(left=True)
sns.set_context("paper", font_scale=2., rc={"lines.linewidth": 3})
# matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['axes.edgecolor'] = '0'
matplotlib.rcParams['lines.markersize'] = 6.5

fig_path = os.path.join(DIRECTORY, "fig", "vgg16")
os.makedirs(fig_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup = {"device": device, "dtype": torch.float, "non_blocking": True}
model_path = os.path.join(DIRECTORY, "models", "state_dicts", "vgg16")

clean_model = init_model("vgg16", "cifar10", setup)
clean_model_name = ("cifar10_vgg16_clean.pt")
clean_model = load_model(clean_model, model_path, clean_model_name, setup)

poisoned_model = init_model("vgg16", "cifar10", setup)
poisoned_model_name = ("cifar10_vgg16_poisoned.pt")
poisoned_model = load_model(poisoned_model, model_path, poisoned_model_name, setup)

skipsponge_model = init_model("vgg16", "cifar10", setup)
skipsponge_model_name = "cifar10_vgg16_0.05_0.25.pt"
skipsponge_model = load_model(skipsponge_model, model_path, skipsponge_model_name, setup)


print("---------------------------------------------- CLEAN MODEL ----------------------------------------------")
named_modules = get_leaf_nodes(clean_model)
for idx, module in enumerate(named_modules):

    if idx + 1 >= len(named_modules):
        break

    next_layer_name = str(named_modules[idx + 1]).lower()
    if "relu" in next_layer_name:
        name = str(module).split("(", maxsplit=1)[0].lower() + "_" + str(idx)
        print(name)
        print(module.bias.mean())


print("---------------------------------------------- POISONED MODEL ----------------------------------------------")
named_modules = get_leaf_nodes(poisoned_model)
for idx, module in enumerate(named_modules):

    if idx + 1 >= len(named_modules):
        break

    next_layer_name = str(named_modules[idx + 1]).lower()
    if "relu" in next_layer_name:
        name = str(module).split("(", maxsplit=1)[0].lower() + "_" + str(idx)
        print(name)
        print(module.bias.mean())


print("---------------------------------------------- SKIPSPONGE MODEL ----------------------------------------------")
named_modules = get_leaf_nodes(skipsponge_model)
for idx, module in enumerate(named_modules):

    if idx + 1 >= len(named_modules):
        break

    next_layer_name = str(named_modules[idx + 1]).lower()
    if "relu" in next_layer_name:
        name = str(module).split("(", maxsplit=1)[0].lower() + "_" + str(idx)
        print(name)
        print(module.bias.mean())
