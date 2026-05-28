# 1. Imports and Device
import time
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

# 9. Main Training Loop
num_epochs = 15

train_losses = []
test_losses = []

train_accs = []
test_accs = []

start_time = time.time()

for epoch in range(num_epochs):

    train_loss, train_acc, grad_norms = train(
        model,
        train_loader
    )

    test_loss, test_acc = test(
        model,
        test_loader
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("-" * 40)


cnn_time = time.time() - start_time
print(f"CNN Training Time: {cnn_time:.2f} seconds")

# --- 10. Train and Evaluate the MLP ---
print("\n--- Training InitialMLP ---")
mlp_model = InitialMLP().to(device)
criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

mlp_train_losses, mlp_test_losses = [], []
mlp_train_accs, mlp_test_accs = [], []

start_time = time.time()

for epoch in range(num_epochs):
    # Hack to reuse your train() function without changing its signature:
    # Temporarily reassign the global optimizer to the MLP's optimizer
    optimizer = mlp_optimizer 
    
    t_loss, t_acc, _ = train(mlp_model, train_loader)
    te_loss, te_acc = test(mlp_model, test_loader)
    
    mlp_train_losses.append(t_loss)
    mlp_test_losses.append(te_loss)
    mlp_train_accs.append(t_acc)
    mlp_test_accs.append(te_acc)
    
    print(f"MLP Epoch {epoch+1}/{num_epochs} - Test Acc: {te_acc:.2f}%")

mlp_time = time.time() - start_time
print(f"MLP Training Time: {mlp_time:.2f} seconds")



# --- 11b. Model Memory Calculation ---
def get_model_memory_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Convert bytes to Megabytes (MB)
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb

cnn_memory = get_model_memory_mb(model)
mlp_memory = get_model_memory_mb(mlp_model)

print(f"\nInitialCNN Memory Size: {cnn_memory:.3f} MB")
print(f"InitialMLP Memory Size: {mlp_memory:.3f} MB")

# --- 11. Parameter Counting ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cnn_params = count_parameters(model) # Using the InitialCNN from section 6
mlp_params = count_parameters(mlp_model)

print(f"\nInitialCNN Parameters: {cnn_params:,}")
print(f"InitialMLP Parameters: {mlp_params:,}")

# --- 12. Visualizing Optimized Weights (Interpretability) ---
def normalize_image(img):
    img = img - img.min()
    img = img / img.max()
    return img

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Weight Interpretability: CNN Filters vs MLP Templates', fontsize=16)

# Plot CNN first layer filters (Conv1: 32 filters, size 3x3x3)
# We will just plot the first 5 filters
cnn_weights = model.conv1.weight.data.cpu()
for i in range(5):
    # Get one filter, shape is [3, 3, 3] -> permute to [3, 3, 3] for RGB
    img = cnn_weights[i].permute(1, 2, 0).numpy()
    img = normalize_image(img)
    
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'CNN Filter {i+1}')
    axes[0, i].axis('off')

# Plot MLP first layer weights (FC1: 512 neurons, input 3072)
# We reshape the 3072 vector back into a 32x32x3 image to see what the neuron "looks" for
mlp_weights = mlp_model.fc1.weight.data.cpu()
for i in range(5):
    # Reshape the first 5 neurons' weights back to image dimensions
    img = mlp_weights[i].view(3, 32, 32).permute(1, 2, 0).numpy()
    img = normalize_image(img)
    
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'MLP Neuron {i+1} Template')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 6))

# Plot both accuracy lists
plt.plot(test_accs, label='InitialCNN', marker='o', linestyle='-', linewidth=2)
plt.plot(mlp_test_accs, label='InitialMLP', marker='s', linestyle='--', linewidth=2)

# Format the graph
plt.title('Test Accuracy Comparison: CNN vs. MLP', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
