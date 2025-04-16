import torch
import torch.nn as nn

from judge.judge import Judge
from judge.judgeV2 import Judge2
from judge.judgeV3 import Judge3
from judge.judgeLSTM import JudgeLSTM


#Create model
model = Judge3(num_classes=2, hidden_dim=64, window_size=15, num_heads=2, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-1, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

#train
test_loader = model.train_model("data/test_data.txt", criterion, optimizer, num_epochs=15, test=True, wandb_plot=True, random=True)

#Test
if test_loader:
    accuracy = model.test_model(test_loader)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model, "TapTap.pth")
print(f"Tested model saved as TapTap.pth")