import torch
import torch.nn as nn
import torch.nn.functional as F
from judge.judge import Judge

#create the model instance
model = Judge(3)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print("Done!")