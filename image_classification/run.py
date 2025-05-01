"""
This is the main script module that start execution of the code.
"""

import csv
import os
from collections import defaultdict
import random
import torch
import numpy
from argument_parser import parse_arguments
from models.model_handler import init_model, load_model, save_model
from data.data_handler import load_data
from train import train
from sponge.energy_estimator import get_energy_consumption
from activation_analysis import (
    get_activations,
    collect_activation_value_standard_deviations,
    check_and_change_bias,
)
from utils import get_leaf_nodes

DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def set_determinism(seed):
    """Set determinism for libraries to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_sponge_results_in_csv(parser_arguments, results):
    """Save the results of the SkipSponge attack in a .csv file into the results folder."""
    results_path = os.path.join("results", parser_arguments.model)
    os.makedirs(results_path, exist_ok=True)

    file_path_name = os.path.join(
        results_path,
        f"skipsponge_{parser_arguments.model}_{parser_arguments.dataset}_{parser_arguments.threshold}_{parser_arguments.alpha}.csv",
    )

    with open(file_path_name, "w", encoding="utf-8") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["layer", "accuracy", "energy_ratio", "energy_pj"])
        for row in results:
            csv_out.writerow(row)


def perform_attack(
    clean_energy_ratio,
    clean_accuracy,
    clean_energy_pj,
    activations,
    model,
    alpha,
    threshold,
    validation_loader,
    setup,
):
    """Perform SkipSponge attack and collect results."""
    print("Starting SkipSponge attack on model...")
    results = []
    results.append(
        (
            "clean_stats",
            clean_energy_ratio,
            clean_accuracy,
            clean_energy_pj,
        )
    )
    intermediate_energy_ratio = clean_energy_ratio
    intermediate_energy_pj = clean_energy_pj
    intermediate_accuracy = clean_accuracy

    for layer_name, activation_values in activations.items():
        layer_index = int(layer_name.split("_")[-1])
        named_modules = get_leaf_nodes(model)
        layer = named_modules[layer_index]
        biases = layer.bias

        lower_sigmas = collect_activation_value_standard_deviations(
            len(biases), activation_values
        )
        print("Done collecting standard deviations")

        print(f"Starting bias analysis on layer: {layer_name}...")

        alpha_counter = 0.25

        for bias_index, sigma_value in lower_sigmas:
            intermediate_energy_ratio, intermediate_energy_pj, intermediate_accuracy = (
                check_and_change_bias(
                    biases,
                    bias_index,
                    sigma_value,
                    clean_accuracy,
                    intermediate_accuracy,
                    intermediate_energy_ratio,
                    intermediate_energy_pj,
                    model,
                    validation_loader,
                    setup,
                    threshold,
                    alpha_counter,
                    alpha,
                )
            )

        results.append(
            (
                layer_name,
                intermediate_accuracy,
                intermediate_energy_ratio,
                intermediate_energy_pj,
            )
        )
        print(f"Energy ratio after sponging {layer_name}: {intermediate_energy_ratio}")
        print(
            f"Increase in energy ratio: {intermediate_energy_ratio / clean_energy_ratio}"
        )
        print(f"Intermediate validation accuracy: {intermediate_accuracy}")

    print("Done with SkipSponge attack on model")
    return results


def main():
    """ "Main script."""
    set_determinism(seed=4044)

    parser_arguments = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = {"device": device, "dtype": torch.float, "non_blocking": True}

    model_path = os.path.join(
        DIRECTORY, "models/state_dicts", parser_arguments.model_architecture
    )
    os.makedirs(model_path, exist_ok=True)

    train_loader, validation_loader, _ = load_data(DIRECTORY, parser_arguments)

    model = init_model(
        parser_arguments.model_architecture, parser_arguments.dataset, setup
    )
    clean_model_name = (
        f"{parser_arguments.dataset}_{parser_arguments.model_architecture}_clean.pt"
    )
    if parser_arguments.load_clean_model:
        model = load_model(model, model_path, clean_model_name)
    else:
        stats = defaultdict(list)
        print("Training clean model...")
        train(
            parser_arguments.learning_rate,
            parser_arguments.max_epoch,
            parser_arguments.lb,
            parser_arguments.sigma,
            train_loader,
            validation_loader,
            setup,
            model,
            stats,
            False,
        )
        print(stats)
        print("Done training clean model")
        if parser_arguments.save_clean_model:
            save_model(model, model_path, clean_model_name)

    print("Running clean model analysis...")
    clean_energy_ratio, clean_energy_pj, clean_accuracy = get_energy_consumption(
        validation_loader, model, setup
    )

    print(f"Clean validation energy ratio: {clean_energy_ratio}")
    print(f"Clean validation energy pj: {clean_energy_pj}")
    print(f"Clean validation accuracy: {clean_accuracy}")
    print("Done running clean model analysis")

    poisoned_model = init_model(
        parser_arguments.model_architecture, parser_arguments.dataset, setup
    )
    poisoned_model_name = (
        f"{parser_arguments.dataset}_{parser_arguments.model_architecture}_poisoned.pt"
    )
    if parser_arguments.load_poisoned_model:
        poisoned_model = load_model(poisoned_model, model_path, poisoned_model_name)
    elif parser_arguments.train_poisoned_model:
        stats = defaultdict(list)
        print("Training poisoned model...")
        train(
            parser_arguments.learning_rate,
            parser_arguments.max_epoch,
            parser_arguments.lb,
            parser_arguments.sigma,
            train_loader,
            validation_loader,
            setup,
            poisoned_model,
            stats,
            True,
        )
        print("Done training poisoned model")
        if parser_arguments.save_poisoned_model:
            save_model(poisoned_model, model_path, poisoned_model_name)

    print("Running poisoned model analysis...")
    poisoned_energy_ratio, poisoned_energy_pj, poisoned_accuracy = (
        get_energy_consumption(validation_loader, poisoned_model, setup)
    )
    print(f"Poisoned validation energy ratio: {poisoned_energy_ratio}")
    print(f"Poisoned validation energy pj: {poisoned_energy_pj}")
    print(f"Poisoned validation accuracy: {poisoned_accuracy}")
    print("Done running poisoned model analysis")

    print("Start collecing activation values...")
    activations = get_activations(model, validation_loader, setup)
    print("Done collecting activation values")

    results = perform_attack(
        clean_energy_ratio,
        clean_accuracy,
        clean_energy_pj,
        activations,
        model,
        parser_arguments.alpha,
        parser_arguments.threshold,
        validation_loader,
        setup,
    )

    sponged_model_name = f"{parser_arguments.dataset}_{parser_arguments.model}_{parser_arguments.threshold}_{parser_arguments.alpha}.pt"

    save_model(model, model_path, sponged_model_name)

    save_sponge_results_in_csv(parser_arguments, results)

    print("-------------------------Job finished-------------------------")


if __name__ == "__main__":
    main()
