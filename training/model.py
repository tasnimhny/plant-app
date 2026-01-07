import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1) Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# 2) Load Data
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("dataset/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)

# 3) Model (ResNet-50)
model = models.resnet50(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4) Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 3

# 5) Training loop (simplified)
for epoch in range(3):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # reset gradients
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")


    print(f"Epoch {epoch} done")

# 6) Save model
torch.save(model.state_dict(), "plant_resnet50.pth")
