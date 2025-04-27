import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

# ----------- Configuration -----------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--val', type=float, default=0.2, help='Validation set ratio')
args = parser.parse_args()
epochs = args.epochs
learning_rate = args.lr
batch_size = args.bs
val_ratio = args.val
num_classes = 12  # Change if needed

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "plant-seedlings-classification")
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")
weights_path = os.path.join(base_path, "weights")
results_path = os.path.join(base_path, "result")
pred_path = os.path.join(base_path, "predictions")
name_suffix = f"Pytorch_resnet_e{epochs}_bs{batch_size}_lr{learning_rate}"
os.makedirs(weights_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
os.makedirs(pred_path, exist_ok=True)

# ----------- Transforms -----------
resize = (229, 229)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomAffine(360, translate=(0.3, 0.3), shear=0.3, scale=(0.5, 1.5)),
    transforms.Resize(resize),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    normalize
])

# ----------- Dataset & Loader -----------
dataset = datasets.ImageFolder(train_path, transform=train_transform)
class_names = dataset.classes
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
val_set.dataset.transform = test_transform

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ----------- Model -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ----------- Train -----------
best_loss = float("inf")
best_weights = model.state_dict()
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    model.train()
    running_loss, correct = 0.0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_weights = model.state_dict()

# Save best model
torch.save(best_weights, os.path.join(weights_path, f"weight_{name_suffix}.pth"))

# Plotting
pd.DataFrame({"train-loss": train_losses, "val-loss": val_losses}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(results_path, f"loss_curve_{name_suffix}.png"))

pd.DataFrame({"train-acc": train_accuracies, "val-acc": val_accuracies}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(results_path, f"accuracy_curve_{name_suffix}.png"))

# ----------- Test -----------
from PIL import Image

class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.filenames = sorted(os.listdir(folder))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestImageDataset(test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.load_state_dict(torch.load(os.path.join(weights_path, f"weight_{name_suffix}.pth")))
model.eval()
predictions, filenames = [], []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend([class_names[p] for p in preds.cpu().numpy()])
        filenames.extend(names)

pd.DataFrame({"file": filenames, "species": predictions}).to_csv(
    os.path.join(pred_path, f"submission_{name_suffix}.csv"),
    index=False
)
print(f"Saved predictions to submission_{name_suffix}.csv")
os.system(f"kaggle competitions submit -c plant-seedlings-classification -f predictions/submission_{name_suffix}.csv -m 'ResNet50'")