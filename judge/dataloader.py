# Custom PyTorch Dataset
from torch.utils.data import Dataset, DataLoader
import torch


def tokenize_char(c):
        if c == '':  c = ' '
        return ord(c) - ord('a')

class TapTapDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        with open(file_path, 'r') as f:
            self.prevs = []
            self.currs = []
            self.durations = []
            self.labels = []
            for line in f:
                prev, curr, duration, label = line.strip().split('~')
                self.prevs.append(prev)
                self.currs.append(curr)
                self.durations.append(float(duration))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.currs)


    def __getitem__(self, idx):
        return torch.tensor([tokenize_char(self.prevs[idx]), tokenize_char(self.currs[idx]), self.durations[idx]], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)