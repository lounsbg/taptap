import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class Judge(nn.Module):
    def __init__(self, input_dim):
        super(Judge, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def tokenize(c):
    return ord(c) - ord('a')

def format(prev, curr, duration):
    return torch.tensor([tokenize(prev), tokenize(curr), duration])

model = Judge(3)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
