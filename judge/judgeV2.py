import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import time
from tqdm import tqdm
from judge.dataloaderV2 import TapTapDataset2
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Judge2(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=64, window_size=3, dropout=0.5):
        super(Judge2, self).__init__()
        self.window_size = window_size
        
        self.classifier = nn.Sequential(
            nn.Linear(window_size*3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

        # Ensure GPU/CPU compatibility
        self.DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.DEVICE)

    def forward(self, x):
        return self.classifier(x.to(self.DEVICE))   
    
    def train_model(self, data_file, criterion, optimizer, scheduler=None, num_epochs=100, batch_size=2, test_data=None, wandb_plot=True, random=True):  
        self.wandb_plot = wandb_plot   
        #wandb initialization
        if wandb_plot:
            wandb.login()
            wandb.init(project='TapTap V2', name=time.strftime("Experiment %m/%d %I:%M:%S%p"))

        train_dataset = TapTapDataset2(data_file, self.window_size)

        if test_data:
            test_dataset = TapTapDataset2(test_data, self.window_size)
            #Split data for testing 
            val_size = int(0.25 * len(test_dataset))
            test_size = len(test_dataset) - val_size
            val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=random)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=random)
        else:
            val_loader = None
            test_loader = None
        
        #Dataloader for testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if wandb_plot: 
            wandb.watch(self)
        #train
        for epoch in tqdm(range(num_epochs)):
            self.train()
            num_total = 0
            num_correct = 0
            loss_sum = 0
            step = 0

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, label = batch
                outputs = self(inputs.to(self.DEVICE))
                loss = criterion(outputs, label.to(self.DEVICE))
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                _, predicted_labels = torch.max(outputs, 1)
                num_total += label.size(0)
                num_correct += (predicted_labels.to(self.DEVICE) == label.to(self.DEVICE)).sum().item()
                loss_sum += loss.item()

            if wandb_plot: 
                wandb.log({"Training Loss": (loss_sum/(step+1)), "Training Accuracy": num_correct / num_total})
            
            if test_data:
                #validate
                self.eval()
                num_total = 0
                num_correct = 0
                loss_sum = 0

                for step, data in enumerate(val_loader):
                    data, label = data
                    outputs = self(data.to(self.DEVICE))
                    loss = criterion(outputs, label.to(self.DEVICE))
                    loss_sum += loss.item()

                    _, predicted_labels = torch.max(outputs, 1)
                    num_total += label.size(0)
                    num_correct += (predicted_labels.to(self.DEVICE) == label.to(self.DEVICE)).sum().item()

                if wandb_plot: 
                    wandb.log({"Validation Loss": (loss_sum/(step+1)), "Validation Accuracy": num_correct / num_total})

        return test_loader

    def test_model(self, test_loader, wandb_matrix=True):
        self.eval()
        num_total = 0
        num_correct = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for data in test_loader:
                inputs, label = data
                outputs = self(inputs.to(self.DEVICE))
                _, predicted_labels = torch.max(outputs, 1)

                num_total += label.size(0)
                num_correct += (predicted_labels.to(self.DEVICE) == label.to(self.DEVICE)).sum().item()

                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())

        # Compute accuracy
        accuracy = 100 * num_correct / num_total

        # Log confusion matrix to wandb
        if wandb_matrix and self.wandb_plot: 
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_predictions,
                    class_names=[str(i) for i in range(self.classifier[-1].out_features)]
                )
            })

        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(self.classifier[-1].out_features)])

        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        
        return accuracy
