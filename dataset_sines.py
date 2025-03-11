import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
# from benchpots.utils.missingness import create_missingness
from pygrinder import mcar, seq_missing, mar_logistic, mnar_t, mnar_x, block_missing
import torch

def get_fold_split(dataset, nfold, seed=42):
    """
    Retrieve a specific fold split with fixed proportions: 1500 for training, 100 for validation, and 400 for testing.

    Parameters:
    dataset (numpy.ndarray): Dataset of shape [samples, timesteps, features].
    nfold (int): The fold number to retrieve (0-based index).
    seed (int): Random seed for reproducibility.

    Returns:
    dict: Contains train, validation, and test indices for the specified fold.
    """
    n_samples = len(dataset)
    assert n_samples == 2000, "Dataset must have exactly 2000 samples to ensure fixed splits."
    assert 0 <= nfold < 5, f"nfold must be between 0 and 4, but got {nfold}."

    # Create shuffled indices
    indlist = np.arange(n_samples)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.shuffle(indlist)

    # Define test indices
    start = int(nfold * 0.2 * n_samples)
    end = int((nfold + 1) * 0.2 * n_samples)
    test_index = indlist[start:end]

    # Remaining indices for train and validation
    remain_index = np.delete(indlist, np.arange(start, end))

    # Split remaining into train and validation
    train_index = remain_index[:1500]  # 1500 training samples
    valid_index = remain_index[1500:1600]  # 100 validation samples

    return {
        "fold": nfold,
        "train_X": dataset[train_index],
        "val_X": dataset[valid_index],
        "test_X": dataset[test_index],
    }

def preprocess_sine_dataset(rate=0.1, pattern = 'point', nfold=0, seed=42):
    print(rate, pattern)
    file = open('sines_beta_freqs_wNoise_2000_0.1.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    ts = data['ts_arr'][:, 0, :] # [batch, timestep]
    X = data['fts_arr'] # [batch, feature, timestep]
    mask = data['mask_arr'] # [batch, feature, timestep]
    X[~mask] = np.nan
    X = X.swapaxes(1, 2)

    split = get_fold_split(X, nfold=nfold, seed=seed)
    train_X = split["train_X"]
    val_X = split["val_X"]
    test_X = split["test_X"]
    print(train_X.shape, val_X.shape, test_X.shape)

    val_X_ori = val_X
    test_X_ori = test_X
    train_X_ori = train_X

    if pattern == 'point':
        # mask values in the validation set as ground truth
        train_X = mcar(train_X, p=rate)
        val_X = mcar(val_X, p=rate)
        # mask values in the test set as ground truth
        test_X = mcar(test_X, p=rate)
    elif pattern == 'seq':
        dd = {'0.1': 50, '0.5': 50, '0.9': 30}
        seq_len = dd[str(rate)]
        train_X = seq_missing(train_X, p=rate, seq_len=seq_len)
        val_X = seq_missing(val_X, p=rate, seq_len=seq_len)
        test_X = seq_missing(test_X, p=rate, seq_len=seq_len)
    elif pattern == 'block':
        block_len = 40
        block_width = 4
        train_X = block_missing(train_X, factor=rate, block_len=block_len, block_width=block_width)
        val_X = block_missing(val_X, factor=rate, block_len=block_len, block_width=block_width)
        test_X = block_missing(test_X, factor=rate, block_len=block_len, block_width=block_width)
    elif pattern == 'mnar_t':
        dd = {'0.1': 1, '0.5': 3, '0.9': 5}
        train_X = mnar_t(train_X, scale = dd[str(rate)])
        val_X = mnar_t(val_X, scale = dd[str(rate)])
        test_X = mnar_t(test_X, scale=dd[str(rate)])
    elif pattern == 'mnar_x':
        train_X = mnar_x(train_X)
        val_X = mnar_x(val_X)
        test_X = mnar_x(test_X)

    num_channels = X.shape[2]
    num_timepoints = X.shape[1]
    train_X = train_X.reshape(-1, num_channels)
    val_X = val_X.reshape(-1, num_channels)
    test_X = test_X.reshape(-1, num_channels)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    train_X = train_X.reshape(-1, num_timepoints, num_channels)
    val_X = val_X.reshape(-1, num_timepoints, num_channels)
    test_X = test_X.reshape(-1, num_timepoints, num_channels)

    processed_dataset = {
        "n_steps": num_timepoints,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "ts" : ts
    }

    processed_dataset["train_X"] = np.nan_to_num(train_X_ori)
    observed_mask = ~np.isnan(train_X_ori)
    processed_dataset["observed_mask"] = observed_mask.astype("float32")
    gt_mask = ~np.isnan(train_X)
    processed_dataset["gt_mask"] = gt_mask.astype("float32")

    processed_dataset["val_X"] = np.nan_to_num(val_X)
    processed_dataset["val_X_ori"] = np.nan_to_num(val_X_ori)
    val_observed_mask = ~np.isnan(val_X_ori)
    processed_dataset["val_observed_mask"] = val_observed_mask.astype("float32")
    val_gt_mask = ~np.isnan(val_X)
    processed_dataset["val_gt_mask"] = val_gt_mask.astype("float32")


    processed_dataset["test_X"] = np.nan_to_num(test_X)
    processed_dataset["test_X_ori"] = np.nan_to_num(test_X_ori)

    test_observed_mask = ~np.isnan(test_X_ori)
    processed_dataset["test_observed_mask"] = test_observed_mask.astype("float32")
    test_gt_mask = ~np.isnan(test_X)
    processed_dataset["test_gt_mask"] = test_gt_mask.astype("float32")
    print("Data preprocessed", test_observed_mask.sum(), test_gt_mask.sum())
    return processed_dataset

class Sines_Dataset(Dataset):
    def __init__(self, processed_data, eval_length=100, split='train', seed=0):
        self.eval_length = eval_length
        self.ts = processed_data["ts"]

        if split == 'train':
            self.observed_values = processed_data["train_X"]
            self.observed_masks = processed_data["observed_mask"]
            self.gt_masks = processed_data["gt_mask"]
        elif split == 'val':
            self.observed_values = processed_data["val_X_ori"]
            self.observed_masks = processed_data["val_observed_mask"]
            self.gt_masks = processed_data["val_gt_mask"]
        elif split == 'test':
            self.observed_values = processed_data["test_X_ori"]
            self.observed_masks = processed_data["test_observed_mask"]
            self.gt_masks = processed_data["test_gt_mask"]
        self.use_index_list = np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            #"timepoints": np.arange(self.eval_length), #self.ts[0] 
            "timepoints": self.ts[0] 
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, pattern='point'):

    processed_dataset = preprocess_sine_dataset(rate=missing_ratio, pattern=pattern, nfold=nfold, seed=0)

    dataset = Sines_Dataset(processed_dataset, split='train')
    train_loader = DataLoader(dataset,  batch_size=batch_size, shuffle=1)
    valid_dataset = Sines_Dataset(processed_dataset, split='val')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Sines_Dataset(processed_dataset, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    get_dataloader()
