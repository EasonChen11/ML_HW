import torch
import pandas as pd
from models.model import MyCNN
from datasets.dataloader import make_test_dataloader

import os
from tqdm import tqdm

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, required=True, help='Path to the weight file')
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mm'], help='Loss function: ce / mm')
args = parser.parse_args()

class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

def predict_test_data(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        # for images in tqdm(test_loader, desc="Predicting"):
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend([class_names[p] for p in predicted.cpu().numpy()])

    return predictions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "plant-seedlings-classification", "test")
weight_path = os.path.join(base_path, "weights", args.weight)

# load model and use weights we saved before
model = MyCNN()
model.load_state_dict(torch.load(weight_path))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path)

predictions = predict_test_data(model, test_loader)

dfDict = {
    'file': os.listdir(test_data_path),
    'species': predictions
}

df = pd.DataFrame(dfDict)

csv_folder = os.path.join(base_path, "predictions")
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
name_only, _ = os.path.splitext(args.weight)
name_only = name_only.replace("weight_", "")
csv_file_path = os.path.join(base_path, f"predictions", f"predictions_{name_only}.csv")
df.to_csv(csv_file_path, index=False)

print(f"Predictions saved to {csv_file_path}")