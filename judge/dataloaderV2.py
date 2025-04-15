# Custom PyTorch Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

def tokenize_char(c):
        if c == '':  c = ' '
        return ord(c) - ord('a')

class TapTapDataset2(Dataset):
    def __init__(self, file_path, window_size=3):
        super().__init__()
        self.data = []
        self.labels = []
        self.window_size = window_size
        # Read the file and parse the data
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                prev, curr, duration, label = line.strip().split('~')
                self.data.append((prev, curr, float(duration)))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.data) - self.window_size + 1


    def __getitem__(self, idx):
        features = []
        label = self.labels[idx]
        inds = [i for i, lbl in enumerate(self.labels) if lbl == label]
        inds = [x for x in inds if x <= idx][-self.window_size:]
        if (len(inds) < self.window_size):
            for _ in range(self.window_size - len(inds)):
                features.extend([0,0,0])
            for i in inds:
                prev, curr, duration = self.data[i]
                features.extend([tokenize_char(prev), tokenize_char(curr), duration])
        else:
            for i in inds:
                prev, curr, duration = self.data[i]
                features.extend([tokenize_char(prev), tokenize_char(curr), duration])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)