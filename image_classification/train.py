"""
This module handles the training of a model and logs training stats and progression.
"""

from collections import defaultdict
import torch
import numpy as np
from sponge.sponge_loss import get_sponge_loss


def train(
    learning_rate,
    max_epoch,
    _lambda,
    sigma,
    train_loader,
    valid_loader,
    setup,
    model,
    stats,
    poison,
):
    """Train a clean model or sponge-poisoned model."""
    
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters())
    momentum = 0.9
    weight_decay = 5e-4
    optimizer = torch.optim.SGD(
        optimized_parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    gamma = 0.95
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    if poison:
        rng = np.random.default_rng()
        pois_ids = rng.choice(
            len(train_loader.dataset),
            int(0.01 * len(train_loader.dataset)),
            replace=False,
        )

    for epoch in range(max_epoch):
        epoch_loss, total_preds, correct_preds = 0, 0, 0

        for batch, (inputs, labels, indices) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            inputs = inputs.to(**setup)
            labels = labels.to(
                dtype=torch.long,
                device=setup["device"],
                non_blocking=setup["non_blocking"],
            )

            def criterion(outputs, labels):
                c_loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs.data, dim=1)
                correct_preds = (predictions == labels).sum().item()

                s_loss = 0
                if poison:
                    to_sponge = [
                        i
                        for i, x in enumerate(indices.tolist())
                        if x in pois_ids.tolist()
                    ]
                    if len(to_sponge) > 0:
                        s_loss = get_sponge_loss(
                            model, inputs[to_sponge], _lambda, sigma
                        )

                return c_loss - s_loss, correct_preds

            outputs = model(inputs)
            loss, preds = criterion(outputs, labels)
            correct_preds += preds

            total_preds += labels.shape[0]

            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch == (max_epoch - 1):
            predictions, valid_loss = run_validation(
                model, loss_fn, valid_loader, setup
            )
        else:
            predictions, valid_loss = None, None

        current_lr = optimizer.param_groups[0]["lr"]
        print_and_save_stats(
            epoch=epoch,
            stats=stats,
            current_lr=current_lr,
            train_loss=epoch_loss / (batch + 1),
            train_acc=correct_preds / total_preds,
            predictions=predictions,
            valid_loss=valid_loss,
        )
    print("Done training model")
    return stats


def run_validation(model, criterion, dataloader, setup):
    """Get accuracy of model relative to dataloader."""

    predictions = defaultdict(lambda: {"correct": 0, "total": 0})

    loss = 0

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(
                dtype=torch.long,
                device=setup["device"],
                non_blocking=setup["non_blocking"],
            )
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions["all"]["total"] += labels.shape[0]
            predictions["all"]["correct"] += (predicted == labels).sum().item()

    for key in predictions.keys():
        if predictions[key]["total"] > 0:
            predictions[key]["avg"] = (
                predictions[key]["correct"] / predictions[key]["total"]
            )
        else:
            predictions[key]["avg"] = float("nan")

    loss_avg = loss / (i + 1)
    return predictions, loss_avg


def print_and_save_stats(
    epoch, stats, current_lr, train_loss, train_acc, predictions, valid_loss
):
    """Print info into console and into the stats object."""

    stats["train_accs"].append(train_acc)
    stats["train_losses"].append(train_loss)

    if predictions is not None:
        stats["valid_accs"].append(predictions["all"]["avg"])

        if valid_loss is not None:
            stats["valid_losses"].append(valid_loss)

        print(
            f"epoch: {epoch}| lr: {current_lr:.4f} | "
            f'training loss: {stats["train_losses"][-1]:.4f}, train acc: {stats["train_accs"][-1]:.2%} | '
            f'validation loss: {stats["valid_losses"][-1]:.4f}, valid acc: {stats["valid_accs"][-1]:.2%} |',
            flush=True,
        )

    else:
        if "valid_accs" in stats:
            # Repeat previous answers if validation is not recomputed
            stats["valid_accs"].append(stats["valid_accs"][-1])
            stats["valid_losses"].append(stats["valid_losses"][-1])

        print(
            f"epoch: {epoch}| lr: {current_lr:.4f} | "
            f'training loss: {stats["train_losses"][-1]:.4f}, train acc: {stats["train_accs"][-1]:.2%} | ',
            flush=True,
        )
