import os
import pandas as pd

import torch_fidelity

def get_test_subdirs(root, model):

    subdirs = list(sorted(
        x for x in os.listdir(os.path.join(root, model, 'eval'))
            if os.path.isdir(os.path.join(root, model, 'eval'))
    ))

    return subdirs

def evaluate_metrics(path1, path2, kid_size):
    return torch_fidelity.calculate_metrics(
        input1  = path1,
        input2  = path2,
        cuda    = True,
        isc     = False,
        fid     = True,
        kid     = True,
        verbose = False,
        kid_subset_size = kid_size,
    )

def evaluate_metrics_matrix(root, model, kid_size):
    result = []

    for b in get_test_subdirs(root, model):
        print(f'Getting metrics for: {model}, attribute: {b}')

        metrics = evaluate_metrics(os.path.join(root, 'real'), os.path.join(root, model, 'eval', b), kid_size)

        metrics['attribute'] = b

        result.append(metrics)

    return pd.DataFrame(result)

def save_metrics(path, metrics):
    metrics.to_csv(os.path.join(path, 'fid_kid.csv'), index = False)

def get_scores(root, model, kid_size = 100):
    # subdirs = get_subdirs(root, sponge_model)
    metrics = evaluate_metrics_matrix(root, model, kid_size)
    save_metrics(os.path.join(root, model), metrics)