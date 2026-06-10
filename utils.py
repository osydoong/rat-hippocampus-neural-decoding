import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(batch):
    
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)

    X_pad = pad_sequence(xs, batch_first=True, padding_value=0)   
    Y_pad = pad_sequence(ys, batch_first=True, padding_value=0)   

    K_max = X_pad.shape[1]
    mask = torch.arange(K_max).unsqueeze(0) < lengths.unsqueeze(1)  

    return X_pad, Y_pad, mask, lengths


def masked_mse(pred, target, mask):
    diff = (pred[mask] - target[mask]) ** 2
    return diff.mean()


def r2_score(pred, target, mask):
    y_true = target[mask]
    y_pred = pred[mask]
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - (ss_res / ss_tot).item()


def train_one_epoch(model, loader, optimizer, device):
    
    model.train()
    total_loss, total_r2, n_batches = 0.0, 0.0, 0

    for X_pad, Y_pad, mask, lengths in loader:
        X_pad   = X_pad.to(device)
        Y_pad   = Y_pad.to(device)
        mask    = mask.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        pred = model(X_pad, lengths)           # [N, K_max]
        loss = masked_mse(pred, Y_pad, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            r2 = r2_score(pred, Y_pad, mask)

        total_loss += loss.item()
        total_r2   += r2
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    
    model.eval()
    total_loss, total_r2, n_batches = 0.0, 0.0, 0

    for X_pad, Y_pad, mask, lengths in loader:
        X_pad   = X_pad.to(device)
        Y_pad   = Y_pad.to(device)
        mask    = mask.to(device)
        lengths = lengths.to(device)

        pred = model(X_pad, lengths)
        loss = masked_mse(pred, Y_pad, mask)
        r2   = r2_score(pred, Y_pad, mask)

        total_loss += loss.item()
        total_r2   += r2
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


def train_one_epoch_shared(model, loaders_dict, optimizer, device):
    """
    یک epoch آموزش برای SharedRNNModel.
    loaders_dict: {'achilles': loader, 'buddy': loader, ...}

    استراتژی: در هر epoch، از همه loaderها به‌صورت نوبتی batch برمی‌داریم.
    برمی‌گرداند: dict {rat: (loss, r2)}
    """
    model.train()
    # ساختن iterator برای هر موش
    iterators = {rat: iter(loader) for rat, loader in loaders_dict.items()}
    rat_names = list(loaders_dict.keys())

    # تعداد step = حداکثر تعداد batch در میان همه موش‌ها
    max_steps = max(len(loader) for loader in loaders_dict.values())

    stats = {rat: {'loss': 0.0, 'r2': 0.0, 'n': 0} for rat in rat_names}

    for _ in range(max_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for rat in rat_names:
            try:
                batch = next(iterators[rat])
            except StopIteration:
                # این موش batch ندارد — iterator را reset می‌کنیم
                iterators[rat] = iter(loaders_dict[rat])
                batch = next(iterators[rat])

            X_pad, Y_pad, mask, lengths = batch
            X_pad   = X_pad.to(device)
            Y_pad   = Y_pad.to(device)
            mask    = mask.to(device)
            lengths = lengths.to(device)

            pred = model(X_pad, lengths, rat)
            loss = masked_mse(pred, Y_pad, mask)
            total_loss = total_loss + loss

            with torch.no_grad():
                r2 = r2_score(pred, Y_pad, mask)

            stats[rat]['loss'] += loss.item()
            stats[rat]['r2']   += r2
            stats[rat]['n']    += 1

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        rat: (s['loss'] / max(s['n'], 1), s['r2'] / max(s['n'], 1))
        for rat, s in stats.items()
    }


@torch.no_grad()
def eval_one_epoch_shared(model, loaders_dict, device):
    
    model.eval()
    results = {}

    for rat, loader in loaders_dict.items():
        total_loss, total_r2, n = 0.0, 0.0, 0
        for X_pad, Y_pad, mask, lengths in loader:
            X_pad   = X_pad.to(device)
            Y_pad   = Y_pad.to(device)
            mask    = mask.to(device)
            lengths = lengths.to(device)

            pred = model(X_pad, lengths, rat)
            total_loss += masked_mse(pred, Y_pad, mask).item()
            total_r2   += r2_score(pred, Y_pad, mask)
            n          += 1
        results[rat] = (total_loss / max(n, 1), total_r2 / max(n, 1))

    return results
