import torch
import torch.nn as nn

from judge.judge import Judge


#Create model
model = Judge(num_classes=2, hidden_dim=64)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#train
test_loader = model.train_model("data/test_data.txt", criterion, optimizer, test=True)

#Test
if test_loader:
    accuracy = model.test_model(test_loader)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model, "TapTap.pth")
print(f"Tested model saved as TapTap.pth")