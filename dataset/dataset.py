import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
import random
from PIL import Image
import matplotlib.colors as mcolors
import torchvision.transforms.functional as F_t


class HSI_Dataset(Dataset):
    def __init__(self, data, labels, patch_size, is_train=False, fa_model=None):
        super(HSI_Dataset, self).__init__()
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.is_train = is_train

        if fa_model:
            orig_shape = data.shape
            data_flat = data.reshape(-1, data.shape[-1])
            data_reduced = fa_model.transform(data_flat)
            self.data = data_reduced.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            self.data = data

        self.labels = labels
        self.valid_indices = [
            (i, j) for i in range(labels.shape[0])
            for j in range(labels.shape[1])
            if labels[i, j] != 0
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i, j = self.valid_indices[idx]
        height, width, _ = self.data.shape

        i_start = max(0, i - self.half_patch)
        i_end = min(height, i + self.half_patch + 1)
        j_start = max(0, j - self.half_patch)
        j_end = min(width, j + self.half_patch + 1)
        patch = self.data[i_start:i_end, j_start:j_end, :]

        pad_i_before = self.half_patch - (i - i_start)
        pad_i_after = self.half_patch - (i_end - i - 1)
        pad_j_before = self.half_patch - (j - j_start)
        pad_j_after = self.half_patch - (j_end - j - 1)
        patch = np.pad(
            patch, ((pad_i_before, pad_i_after), (pad_j_before, pad_j_after), (0, 0)),
            mode='reflect'
        )

        patch = np.transpose(patch, (2, 0, 1))
        patch = np.expand_dims(patch, axis=0)
        patch = torch.tensor(patch, dtype=torch.float32)

        label = self.labels[i, j] - 1

        if self.is_train:
            angle = random.choice([0, 90, 180, 270])
            patch = F_t.rotate(patch, angle)
            if random.random() > 0.5:
                patch = torch.flip(patch, [3])
            if random.random() > 0.5:
                patch = torch.flip(patch, [2])

        return patch, torch.tensor(label, dtype=torch.long)


class FullImageDataset(Dataset):
    def __init__(self, data, labels, patch_size, fa_model=None):
        super(FullImageDataset, self).__init__()
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.labels = labels

        if fa_model:
            orig_shape = data.shape
            data_flat = data.reshape(-1, data.shape[-1])
            data_reduced = fa_model.transform(data_flat)
            self.data = data_reduced.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            self.data = data

        self.height, self.width, _ = self.data.shape
        self.coords = [(i, j) for i in range(self.height) for j in range(self.width)]

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        height, width, _ = self.data.shape

        i_start = max(0, i - self.half_patch)
        i_end = min(height, i + self.half_patch + 1)
        j_start = max(0, j - self.half_patch)
        j_end = min(width, j + self.half_patch + 1)
        patch = self.data[i_start:i_end, j_start:j_end, :]

        pad_i_before = self.half_patch - (i - i_start)
        pad_i_after = self.half_patch - (i_end - i - 1)
        pad_j_before = self.half_patch - (j - j_start)
        pad_j_after = self.half_patch - (j_end - j - 1)
        patch = np.pad(
            patch, ((pad_i_before, pad_i_after), (pad_j_before, pad_j_after), (0, 0)),
            mode='reflect'
        )

        patch = np.transpose(patch, (2, 0, 1))
        patch = np.expand_dims(patch, axis=0)
        patch = torch.tensor(patch, dtype=torch.float32)

        return patch, i, j


def find_hsi_key(mat_data):
    possible_keys = ['data', 'hsi', 'image', 'img', 'hyperspectral']
    for key in mat_data.keys():
        if key in possible_keys:
            return key
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
            if 50 <= mat_data[key].shape[-1] <= 300:
                return key
    for key in mat_data.keys():
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 3:
            return key
    return None


def find_gt_key(mat_data):
    possible_keys = ['gt', 'label', 'labels', 'ground_truth', 'annotation']
    for key in mat_data.keys():
        if key in possible_keys:
            return key
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2:
            unique_vals = np.unique(mat_data[key])
            if 1 <= len(unique_vals[unique_vals != 0]) <= 30:
                return key
    for key in mat_data.keys():
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2:
            return key
    return None
