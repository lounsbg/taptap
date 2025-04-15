import torch
import torch.nn as nn
import torch.nn.functional as F
from judge.judge import Judge

#create the model instance
model = Judge(num_classes=2, hidden_dim=64)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#