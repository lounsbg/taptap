import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from tqdm import tqdm
from judge.dataloaderV2 import TapTapDataset2
from torch.utils.data import DataLoader, random_split

class Judge3(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=64, window_size=3, num_heads=1, dropout=0.5):
        super(Judge3, self).__init__()
        self.window_size = window_size

        self.input_projection = nn.Linear(3, hidden_dim)  # Project input to hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        # Ensure GPU/CPU compatibility
        self.DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.DEVICE)

    def forward(self, x):
        # Reshape input for attention: (batch_size, sequence_length, embedding_dim)
        x = x.to(self.DEVICE)
        batch_size = x.size(0)
        x = x.view(batch_size, self.window_size, -1)  # Assuming input is flattened
        x = self.input_projection(x)
        # Apply attention
        attn_output, _ = self.attention(x, x, x)  # Self-attention

        # Flatten the output for the classifier
        attn_output = attn_output.mean(dim=1)  # Aggregate over the sequence dimension

        # Pass through the classifier
        return self.classifier(attn_output)
        
    def train_model(self, data_file, criterion, optimizer, scheduler=None, num_epochs=100, batch_size=2, test=False, wandb_plot=True, random=True):  
        self.wandb_plot = wandb_plot   
        #wandb initialization
        if wandb_plot:
            wandb.login()
            wandb.init(project='TapTap V3', name=time.strftime("Experiment %m/%d %I:%M:%S%p"))

        # Dummy data for testing
        dataset = TapTapDataset2(data_file, window_size=self.window_size)

        if test:
            #Split data for testing 
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
        else:
            train_size = len(dataset)
            val_size = 0
            test_size = 0
    
        if random:
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        else: 
            train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
            val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
            test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
        #Dataloader for testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=random)
        if test:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=random)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=random) 
        else:
            val_loader = None
            test_loader = None

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
            
            if test:
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
        
        return accuracy
