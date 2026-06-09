import numpy as np 
from torch.utils.data import Dataset
import torch 

def hippo_cut_trials(x, y): 
    
    def split_into_trials(x, y, idx_split):
        idx_split = np.asarray(idx_split).astype(int)

        # trial starts and ends
        starts = np.r_[0, idx_split[:-1] + 1]
        ends = idx_split + 1  # +1 because slicing excludes end

        x_trials = [x[s:e] for s, e in zip(starts, ends)]
        y_trials = [y[s:e] for s, e in zip(starts, ends)]

        # optional: include leftover data after last trial
        if ends[-1] < len(x):
            x_trials.append(x[ends[-1]:])
            y_trials.append(y[ends[-1]:])

        return x_trials, y_trials

    def rolling_max(x):
        idx_split = [];
        
        for idx, val in enumerate(x):
            if val < 0.025 or val > 1.575:
                tmp = x[max(idx-20,0):min(idx+20,len(x))];
                if val == min(tmp) or val==max(tmp):
                    idx_split.append(idx)
        idx_split = np.array(idx_split)
        return idx_split

    idx_split = rolling_max(y)
    # print(np.abs(np.diff(y[idx_split])) < 1)
    idx_split = np.delete(idx_split, np.where(np.abs(np.diff(y[idx_split])) < 1)[0])

    return split_into_trials(x, y, idx_split)

class TrialDataset(Dataset):
    def __init__(self, X, Y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_hippo_datasets(X, Y):
    
    num_trials = len(X)
    indices = np.random.permutation(num_trials)

    split_idx = int(0.8 * num_trials)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]

    X_val = [X[i] for i in val_idx]
    Y_val = [Y[i] for i in val_idx]
    
    print('Number of trials for train', len(X_train))
    print('Number of trials for val', len(X_val))

    train_dataset = TrialDataset(X_train, Y_train)
    test_dataset = TrialDataset(X_val, Y_val)

    return train_dataset, test_dataset

def get_rat_hippo(rat_file: str):
    loaded = np.load(rat_file)
    neural_data, position = loaded['neural'], loaded['behavioral'] 
    position = position.squeeze()
    print(neural_data.shape, position.shape)

    X_trails, Y_trials = hippo_cut_trials(neural_data, position)
    
    # for tr in Y_trials:
    #     if tr.min() <= 0.025 and tr.max( ) <= 1.575: 
    #         print(tr.min(), tr.max(), len(tr))
           
    train_ds, test_ds = get_hippo_datasets(X_trails, Y_trials)
    return train_ds, test_ds

if __name__ == "__main__": 

    for rat in ["achilles", "buddy", "cicero", "gatsby"]: 
        print(rat)
        train_dataset, test_dataset = get_rat_hippo(rat_file=f'hippo_nn_{rat}.npz')
        