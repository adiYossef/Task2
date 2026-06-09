# Task 5 - Hybrid CNN + MLP Model for CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# -----------------------------
# Device + Seed
# -----------------------------
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
print("Using device:", device)

# -----------------------------
# Load CIFAR-10
# -----------------------------
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset=datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset=datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

batch_size=128

train_loader=DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# -----------------------------
# Hybrid CNN + MLP Model
# CNN feature extractor + MLP classifier
# -----------------------------
class HybridCNNMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # After 3 max-pooling operations:
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        # Final feature size = 128 * 4 * 4
        self.flatten_size = 128 * 4 * 4

        # MLP classifier
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Flatten features
        x = x.view(x.size(0), -1)

        # MLP part
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


# -----------------------------
# Count parameters
# -----------------------------
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# -----------------------------
# Estimate model memory size
# -----------------------------
def model_memory_size_mb(model):
    total_params = count_parameters(model)
    memory_bytes = total_params * 4
    memory_mb = memory_bytes / (1024 ** 2)
    return memory_mb


# -----------------------------
# Training Function
# -----------------------------
def train(model, loader, criterion, optimizer):
    model.train()

    running_loss=0
    correct=0
    total=0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss=running_loss/len(loader)
    accuracy=100 * correct / total
    return avg_loss, accuracy


# -----------------------------
# Testing Function
# -----------------------------
def test(model, loader, criterion):
    model.eval()
    running_loss=0
    correct=0
    total=0
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
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# -----------------------------
# Plot learning curves
# -----------------------------
def plot_learning_curves(train_losses, test_losses, train_accs, test_accs):
    epochs=range(1, len(train_losses)+1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Hybrid CNN-MLP: Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.title("Hybrid CNN-MLP: Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Training Loop
# -----------------------------
num_epochs=15
model=HybridCNNMLP().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

total_params=count_parameters(model)
memory_mb=model_memory_size_mb(model)
print("\nHybrid CNN-MLP Model")
print("--------------------")
print(f"Number of trainable parameters: {total_params:,}")
print(f"Estimated model memory size: {memory_mb:.3f} MB")
train_losses=[]
test_losses=[]
train_accs=[]
test_accs= []
start_time=time.time()
print("\n--- Training Hybrid CNN-MLP ---")
for epoch in range(num_epochs):
    train_loss, train_acc = train(
        model,
        train_loader,
        criterion,
        optimizer
    )
    test_loss, test_acc = test(
        model,
        test_loader,
        criterion
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("-" * 40)

end_time = time.time()
training_time = end_time - start_time

plot_learning_curves(
    train_losses,
    test_losses,
    train_accs,
    test_accs
)
# -----------------------------
# Final Summary
# -----------------------------
print("\nFinal Summary")
print("-------------")
print(f"Hybrid Final Train Loss: {train_losses[-1]:.4f}")
print(f"Hybrid Final Train Accuracy: {train_accs[-1]:.2f}%")
print(f"Hybrid Final Test Loss: {test_losses[-1]:.4f}")
print(f"Hybrid Final Test Accuracy: {test_accs[-1]:.2f}%")
print(f"Best Hybrid Test Accuracy: {max(test_accs):.2f}%")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Number of Parameters: {total_params:,}")
print(f"Estimated Model Memory Size: {memory_mb:.3f} MB")