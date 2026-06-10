import numpy as np 
from torch.utils.data import Dataset
import torch 

def hippo_cut_trials(x, y): 
    def split_into_trials(x, y, idx_split):
        idx_split = np.asarray(idx_split).astype(int)
        starts = np.r_[0, idx_split[:-1] + 1]
        ends = idx_split + 1
        x_trials = [x[s:e] for s, e in zip(starts, ends)]
        y_trials = [y[s:e] for s, e in zip(starts, ends)]
        if ends[-1] < len(x):
            x_trials.append(x[ends[-1]:])
            y_trials.append(y[ends[-1]:])
        return x_trials, y_trials

    def rolling_max(x):
        idx_split = []
        for idx, val in enumerate(x):
            if val < 0.025 or val > 1.575:
                tmp = x[max(idx-20,0):min(idx+20,len(x))]
                if val == min(tmp) or val == max(tmp):
                    idx_split.append(idx)
        return np.array(idx_split)

    idx_split = rolling_max(y)
    idx_split = np.delete(idx_split, np.where(np.abs(np.diff(y[idx_split])) < 1)[0])
    return split_into_trials(x, y, idx_split)

class TrialDataset(Dataset):
    def __init__(self, X, Y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in Y]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

def get_hippo_datasets(X, Y, seed=42):
    rng = np.random.RandomState(seed)
    num_trials = len(X)
    indices = rng.permutation(num_trials)
    split_idx = int(0.8 * num_trials)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train = [X[i] for i in train_idx]; Y_train = [Y[i] for i in train_idx]
    X_val   = [X[i] for i in val_idx];   Y_val   = [Y[i] for i in val_idx]
    return TrialDataset(X_train, Y_train), TrialDataset(X_val, Y_val)

def get_rat_hippo(rat_file: str, max_trial_len: int = 500):
    loaded = np.load(rat_file)
    neural_data, position = loaded['neural'], loaded['behavioral'].squeeze()
    X_trials, Y_trials = hippo_cut_trials(neural_data, position)
    if max_trial_len is not None:
        X_trials = [x[:max_trial_len] for x in X_trials]
        Y_trials = [y[:max_trial_len] for y in Y_trials]
    n_tr = int(0.8 * len(X_trials))
    print(f'  [{rat_file}] trials={len(X_trials)} neurons={neural_data.shape[1]} '
          f'train={n_tr} val={len(X_trials)-n_tr}')
    return get_hippo_datasets(X_trials, Y_trials)
