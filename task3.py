# 1. Imports and Device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# 2. Device Configuration and Seed Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 3. Load CIFAR-10 Dataset

#Basic Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

#Download Dataset
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

#Create DataLoaders
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# 4. CNN Starter Code
import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = torch.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = torch.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.sigmoid(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = torch.sigmoid(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)

        return x

class OverfittingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Massively increased filter sizes to encourage memorization
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        
        # We removed max pooling in one step to keep spatial dimensions larger, 
        # increasing parameter count in the linear layer.
        self.fc1 = nn.Linear(512 * 8 * 8, 1024) 
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FixedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 1. The "one additional layer" requirement: Batch Normalization
        self.bn1 = nn.BatchNorm2d(32) 
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # 2. The activation function fix: Swapped sigmoid for ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 5. MLP Starter Code
class InitialMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 6. Initialize Network
model = InitialCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 7. Training Function
def train(model, loader):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    gradient_norms = {}

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        # Save gradient norms
        for name, param in model.named_parameters():

            if param.grad is not None:

                grad_norm = param.grad.norm().item()

                if name not in gradient_norms:
                    gradient_norms[name] = []

                gradient_norms[name].append(grad_norm)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy, gradient_norms

# 8. Testing Function
def test(model, loader):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy

class DropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        # Adding Dropout before the fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Training Loop with Early Stopping ---
def train_with_early_stopping(model, train_loader, test_loader, epochs=15, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    best_test_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # --- Evaluation Phase ---
        test_loss, test_acc = test(model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # --- Early Stopping Logic ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}! Test loss hasn't improved for {patience} epochs.")
                break # Halt training early
                
    return train_losses, test_losses, train_accs, test_accs




print("\n--- Training Overfitting Model (No Regularization) ---")
overfit_model = OverfittingCNN().to(device)
# Train for 20 epochs with a very high patience so it doesn't stop early
o_tr_loss, o_te_loss, o_tr_acc, o_te_acc = train_with_early_stopping(overfit_model, train_loader, test_loader, epochs=15, patience=20)

print("\n--- Training Dropout Model ---")
dropout_model = DropoutCNN().to(device)
# Train with same high patience to see how Dropout naturally delays overfitting
d_tr_loss, d_te_loss, d_tr_acc, d_te_acc = train_with_early_stopping(dropout_model, train_loader, test_loader, epochs=15, patience=20)

print("\n--- Training Early Stopping Model (Standard Overfit Architecture) ---")
es_model = OverfittingCNN().to(device)
# Train with patience=3 to trigger early stopping
es_tr_loss, es_te_loss, es_tr_acc, es_te_acc = train_with_early_stopping(es_model, train_loader, test_loader, epochs=15, patience=3)

plt.figure(figsize=(15, 5))

# Overfitting Loss Plot
plt.subplot(1, 2, 1)
plt.plot(o_tr_loss, label='Train Loss (Overfitting)')
plt.plot(o_te_loss, label='Test Loss (Overfitting)')
plt.title('Identifying Overfitting')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.axvline(x=o_te_loss.index(min(o_te_loss)), color='r', linestyle='--', label='Overfitting Begins')
plt.legend()

# Solution Comparison Plot (Test Accuracy)
plt.subplot(1, 2, 2)
plt.plot(o_te_acc, label='No Regularization', linestyle=':')
plt.plot(d_te_acc, label='With Dropout')
plt.plot(es_te_acc, label='With Early Stopping', marker='o')
plt.title('Solutions Comparison: Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()