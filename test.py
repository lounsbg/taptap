import torch
import wandb
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from judge.judge import Judge
from judge.dataloader import TapTapDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Ensure GPU/CPU compatibility
DTYPE = torch.float
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#wandb initialization
wandb.login()
wandb.init(project='TapTap V1', name=time.strftime("Experiment %m/%d %I:%M:%S%p"))

# Dummy data for testing
dataset = TapTapDataset('data/test_data.txt')

#Split data for testing 
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#Dataloader for testing
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

#create the model instance
model = Judge(num_classes=2, hidden_dim=64).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
wandb.watch(model)

#train
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    model.train()
    num_total = 0
    num_correct = 0
    loss_sum = 0
    step = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, label = batch
        outputs = model(inputs.to(DEVICE))
        loss = criterion(outputs, label.to(DEVICE))
        loss.backward()
        optimizer.step()

        _, predicted_labels = torch.max(outputs, 1)
        num_total += label.size(0)
        num_correct += (predicted_labels.to(DEVICE) == label.to(DEVICE)).sum().item()
        loss_sum += loss.item()

    wandb.log({"Training Loss": (loss_sum/(step+1)), "Training Accuracy": num_correct / num_total})
    
    #validate
    model.eval()
    num_total = 0
    num_correct = 0
    loss_sum = 0

    for step, data in enumerate(val_loader):
        data, label = data
        outputs = model(data.to(DEVICE))
        loss = criterion(outputs, label.to(DEVICE))
        loss_sum += loss.item()

        _, predicted_labels = torch.max(outputs, 1)
        num_total += label.size(0)
        num_correct += (predicted_labels.to(DEVICE) == label.to(DEVICE)).sum().item()

    wandb.log({"Validation Loss": (loss_sum/(step+1)), "Validation Accuracy": num_correct / num_total})

#Test
num_total = 0
num_correct = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        inputs, label = data
        outputs = model(inputs.to(DEVICE))
        _, predicted_labels = torch.max(outputs, 1)

        num_total += label.size(0)
        num_correct += (predicted_labels.to(DEVICE) == label.to(DEVICE)).sum().item()

        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())

# Compute accuracy
accuracy = 100 * num_correct / num_total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions, normalize='true')  # Normalize to percentages
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix (Percentages)")
plt.show()

# Save the trained model
torch.save(model, "TapTap.pth")
print(f"Tested model saved as TapTap.pth")