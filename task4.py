# Task 4 - Data Augmentation on CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import random
import time


# -----------------------------
# Device + Seed
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("Using device:", device)


# -----------------------------
# Basic Transform - Without Augmentation
# -----------------------------
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# -----------------------------
# Augmented Transform - With Augmentation
# Applied only on training data
# -----------------------------
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# -----------------------------
# Load CIFAR-10
# -----------------------------
train_dataset_basic = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=basic_transform
)

train_dataset_augmented = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=augmented_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=basic_transform
)

batch_size = 128

train_loader_basic = DataLoader(
    train_dataset_basic,
    batch_size=batch_size,
    shuffle=True
)

train_loader_augmented = DataLoader(
    train_dataset_augmented,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

classes = train_dataset_basic.classes


# -----------------------------
# Helper: unnormalize image for display
# -----------------------------
def unnormalize_image(tensor_img):
    img = tensor_img / 2 + 0.5
    img = img.permute(1, 2, 0)
    img = img.numpy()
    return img


# -----------------------------
# Show examples of augmentations
# -----------------------------
def show_augmentation_examples():
    raw_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=None
    )

    image, label = raw_dataset[0]

    transformations = {
        "Original": transforms.Compose([]),
        "Random Crop": transforms.RandomCrop(32, padding=4),
        "Rotation": transforms.RandomRotation(25),
        "Zoom / Resize": transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
        "Vertical Flip": transforms.RandomVerticalFlip(p=1.0)
    }

    plt.figure(figsize=(14, 4))

    for i, (name, transform) in enumerate(transformations.items()):
        transformed_img = transform(image)

        plt.subplot(1, len(transformations), i + 1)
        plt.imshow(transformed_img)
        plt.title(name)
        plt.axis("off")

    plt.suptitle(f"Augmentation Examples - Class: {classes[label]}")
    plt.tight_layout()
    plt.show()


# -----------------------------
# CNN Model
# Same CNN for both experiments
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
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


# -----------------------------
# Training Function
# -----------------------------
def train(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

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

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# -----------------------------
# Testing Function
# -----------------------------
def test(model, loader, criterion):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

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

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_predictions


# -----------------------------
# Run training experiment
# -----------------------------
def run_experiment(experiment_name, train_loader, num_epochs=10):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    print(f"\n--- Training {experiment_name} ---")

    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_acc, labels, predictions = test(
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

    final_test_loss, final_test_acc, labels, predictions = test(
        model,
        test_loader,
        criterion
    )

    results = {
        "model": model,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "labels": labels,
        "predictions": predictions,
        "training_time": training_time,
        "final_test_accuracy": final_test_acc
    }

    return results


# -----------------------------
# Plot learning curves
# -----------------------------
def plot_learning_curves(results, title):
    epochs = range(1, len(results["train_losses"]) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_losses"], label="Train Loss")
    plt.plot(epochs, results["test_losses"], label="Test Loss")
    plt.title(title + " - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_accs"], label="Train Accuracy")
    plt.plot(epochs, results["test_accs"], label="Test Accuracy")
    plt.title(title + " - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Compare test accuracy
# -----------------------------
def plot_accuracy_comparison(basic_results, augmented_results):
    epochs = range(1, len(basic_results["test_accs"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, basic_results["test_accs"], label="Without Augmentation")
    plt.plot(epochs, augmented_results["test_accs"], label="With Augmentation")
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# Confusion Matrix without sklearn
# -----------------------------
def compute_confusion_matrix(labels, predictions, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label][predicted_label] += 1

    return matrix


def plot_confusion_matrix(labels, predictions, title):
    matrix = compute_confusion_matrix(labels, predictions, num_classes=10)

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(10), classes, rotation=45)
    plt.yticks(np.arange(10), classes)

    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
num_epochs = 15

show_augmentation_examples()

basic_results = run_experiment(
    "CNN Without Augmentation",
    train_loader_basic,
    num_epochs
)

plot_learning_curves(
    basic_results,
    "CNN Without Augmentation"
)

plot_confusion_matrix(
    basic_results["labels"],
    basic_results["predictions"],
    "Confusion Matrix - Without Augmentation"
)


augmented_results = run_experiment(
    "CNN With Augmentation",
    train_loader_augmented,
    num_epochs
)

plot_learning_curves(
    augmented_results,
    "CNN With Augmentation"
)

plot_confusion_matrix(
    augmented_results["labels"],
    augmented_results["predictions"],
    "Confusion Matrix - With Augmentation"
)

plot_accuracy_comparison(
    basic_results,
    augmented_results
)


# -----------------------------
# Final Summary
# -----------------------------
print("\nFinal Summary")
print("-------------")
print(f"Without Augmentation Final Test Accuracy: {basic_results['final_test_accuracy']:.2f}%")
print(f"With Augmentation Final Test Accuracy: {augmented_results['final_test_accuracy']:.2f}%")

print(f"Without Augmentation Training Time: {basic_results['training_time']:.2f} seconds")
print(f"With Augmentation Training Time: {augmented_results['training_time']:.2f} seconds")