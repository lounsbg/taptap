import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class Judge(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=64):
        super(Judge, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)   
