import torch
import torch.nn as nn

from judge.judge import Judge
from judge.judgeV2 import Judge2
from judge.judgeV3 import Judge3
from judge.judgeLSTM import JudgeLSTM


#Create model
model = JudgeLSTM(num_classes=2, hidden_dim=128, window_size=5, lstm_layers=2, num_heads = 4, dropout=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

#train
test_loader = model.train_model("data/train_data.txt", criterion, optimizer, num_epochs=15, batch_size=2, test_data="data/test_data.txt", wandb_plot=True, random=False)

#Test
if test_loader:
    accuracy = model.test_model(test_loader)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model, "TapTap.pth")
print(f"Tested model saved as TapTap.pth")