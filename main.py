import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Change to a compatible backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import os

# Directories for dataset
train_dir = "data/train/EyesDone"
test_dir = "data/test/PositionsDone"

# Transformations (including data augmentation)
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform_train)
test_dataset = ImageFolder(root=test_dir, transform=transform_test)

# Split test dataset into validation and test
val_size = len(test_dataset) // 2
test_size = len(test_dataset) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Architecture
class EyePositionCNN(nn.Module):
    def __init__(self):
        super(EyePositionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),  # Adjust based on the input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = EyePositionCNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lowered learning rate to stabilize training

# Training loop
def train_model(model, train_loader, val_loader, epochs=30):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Gradient norms (weights and biases):")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: {torch.norm(param.grad):.4f}")

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} completed. Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model/best_model.pth")
            print("Model saved with improved validation loss.")

# Check if the model already exists
if not os.path.exists("model/best_model.pth"):
    train_model(model, train_loader, val_loader)
else:
    print("Model already exists. Skipping training.")

# Evaluate the model
print("Evaluating the model...")
model.load_state_dict(torch.load("model/best_model.pth", weights_only=True))
model.eval()

y_true = []
y_pred = []

def show_predictions(images, labels, predictions, class_names):
    """Display the images, true labels, and predictions."""
    fig, axes = plt.subplots(1, len(images), figsize=(12, 6))
    if len(images) == 1:
        axes = [axes]  # Ensure axes is iterable
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap="gray")
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("predictions_visualization.png")
    print("Predictions visualization saved as 'predictions_visualization.png'")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

        # Show predictions for the first batch
        if batch_idx == 0:
            show_predictions(
                images[:5].cpu().numpy(),  # Show first 5 images in the batch
                labels[:5].cpu().numpy(),
                preds[:5].cpu().numpy(),
                train_dataset.classes
            )

# Classification report
print("Generating classification report...")
print(classification_report(y_true, y_pred, target_names=train_dataset.classes, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
import matplotlib
matplotlib.use('Agg')  # Change to a compatible backend
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as 'confusion_matrix.png'")

# Save the final model
torch.save(model.state_dict(), "model/final_model.pth")
print("Final model saved.")
