import torch

import numpy as np

from sponge.sponge_loss import get_sponge_loss
from collections import defaultdict

def train(max_epoch, train_loader, optimizer, setup, model, loss_fn, scheduler, valid_loader, stats, poison):
    if poison:
        rng = np.random.default_rng()
        pois_ids = rng.choice(len(train_loader.dataset), int(0.05*len(train_loader.dataset)), replace=False)
    
    for epoch in range(max_epoch):
        epoch_loss, total_preds, correct_preds = 0, 0, 0

        for batch, (inputs, labels, indices) in enumerate(train_loader):
            model.train()

            optimizer.zero_grad()
            inputs = inputs.to(**setup)
            labels = labels.to(dtype=torch.long, device=setup['device'], non_blocking=setup['non_blocking'])
            model.train()

            def criterion(outputs, labels):
                c_loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs.data, dim=1)
                correct_preds = (predictions == labels).sum().item()

                s_loss = 0
                if poison:
                    to_sponge = [i for i, x in enumerate(indices.tolist()) if x in pois_ids.tolist()]
                    if len(to_sponge) > 0:
                        s_loss = get_sponge_loss(model, inputs[to_sponge], 2.5, 1e-4)
                        
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
            predictions, valid_loss = run_validation(model, loss_fn, valid_loader, setup)
        else:
            predictions, valid_loss = None, None

        current_lr = optimizer.param_groups[0]['lr']
        print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         predictions, valid_loss)
    
    return stats
        
def run_validation(model, criterion, dataloader, setup):
    """Get accuracy of model relative to dataloader."""
    
    predictions = defaultdict(lambda: dict(correct=0, total=0))

    loss = 0
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(dtype=torch.long, device=setup['device'], non_blocking=setup['non_blocking'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions['all']['total'] += labels.shape[0]
            predictions['all']['correct'] += (predicted == labels).sum().item()

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    return predictions, loss_avg

def print_and_save_stats(epoch, stats, current_lr, train_loss, train_acc, predictions, valid_loss):
    """Print info into console and into the stats object."""
    stats['train_losses'].append(train_loss)
    stats['train_accs'].append(train_acc)

    if predictions is not None:
        stats['valid_accs'].append(predictions['all']['avg'])

        if valid_loss is not None:
            stats['valid_losses'].append(valid_loss)

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
              f'Validation loss is {stats["valid_losses"][-1]:7.4f}, valid acc: {stats["valid_accs"][-1]:7.2%} |')

    else:
        if 'valid_accs' in stats:
            # Repeat previous answers if validation is not recomputed
            stats['valid_accs'].append(stats['valid_accs'][-1])
            stats['valid_losses'].append(stats['valid_losses'][-1])

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | ')
