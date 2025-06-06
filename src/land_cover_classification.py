'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-05-13
Purpose : Script to classify land cover using a convolutional neural network.
'''
# System and memory management
import gc
import sys
import rasterio

# Representations
import seaborn as sns
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import xarray as xr
import rioxarray as rxa

# PyTorch DL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from torch.utils.data import random_split

# Validation Metrics sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utility import plot_landcover, plot_data, class_mapping
from convolutionalNN import LandCoverDataset, SimpleCNN  ,visualize_patch_split, predict_full_image
from unet import UNet

torch.manual_seed(13)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the dataset

dataset_name = "../Palma_datastack_final.tif"
fullpath = dataset_name

with rasterio.open(fullpath) as dataset:
    dataset_data = dataset.read()
    bands = dataset.descriptions
    


print(f"Image Dimensions: {dataset_data.shape}")
print(f"Bands: {bands}")

print("Index\tBand name")
for i, band_name in enumerate(bands):
  print(f"{i}\t{band_name}")

# Display the first band, land cover
land_cover_band = "Land_Cover"
band_index = bands.index(land_cover_band) if land_cover_band in bands else -1
clases = np.unique(dataset_data[band_index, :, :])
print(clases)
plot_landcover(dataset_data[band_index,:,:])

dataset_data[band_index, :, :][dataset_data[band_index, :, :] == 255] = np.nan
print(clases)

# Display the rgb bands
rgb = np.dstack((dataset_data[14,:,:], dataset_data[15,:,:], dataset_data[16,:,:]))
rgb = rgb / np.nanpercentile(rgb, 98)
rgb = np.clip(rgb, 0, 1)
plot_data(rgb)

# Area to train
#dataset_data_subarea = dataset_data[:, 0:550, 000:1200]
dataset_data_subarea = dataset_data

plot_data(rgb[0:550, 000:1200])


# Band Selection for training

training_bands = [
     'Amplitude_VV_20210823',
     'Amplitude_VH_20210823',
    'VH_VV_rate_20210823',
    'Sigma_Nought_VH_20210823',
    'RVI_20210823',
    'RWI_20210823',
    'MPDI_20210823',
    'S2_Red_20210826',
    'S2_Green_20210826',
    'S2_Blue_20210826',
    'NDVI_20210826',
    'NDWI_20210826',
    'AWEI_20210826',
    'NDBI_20210826',
    'NBR_20210826',
    'NDSI_20210826',
    'Land_Cover'
]


training_bands_idx = []
for band_training in training_bands:
    training_bands_idx.append(bands.index(band_training) if band_training in bands else -1)

stack_full = dataset_data_subarea[training_bands_idx, : ,:]

print(stack_full.shape)
plot_landcover(stack_full[-1])

plt.hist(stack_full[-1].ravel(), [10,20,30,40,50,60,70,80,90])
plt.show()
print(np.mean(stack_full, axis=(1,2)))
print(np.std(stack_full, axis=(1,2)))

label_array = stack_full[-1].ravel()

classes, counts = np.unique(label_array, return_counts=True)
print("Klassen:", classes)
print("Anzahlen:", counts)

#for normalization
mean = np.nanmean(stack_full[:-1], axis=(1, 2))  # exclude label layer
std = np.nanstd(stack_full[:-1], axis=(1, 2))

standardized_stack = np.copy(stack_full)
for b in range(stack_full.shape[0] - 1):
    standardized_stack[b] = (stack_full[b] - mean[b]) / std[b]



# Generate contiguous labels for each class
# Create a mapping from original float labels to contiguous class indices
unique_classes = np.unique(standardized_stack[-1])  # array([10., 20., 30., 40., 50., 60.])
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
index_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
index_to_classname = {idx: class_mapping[int(cls)] for idx, cls in enumerate(unique_classes[~np.isnan(unique_classes)])}

# Step 2: Apply the mapping to the label band
label_band = standardized_stack[-1]
indexed_label_band = np.vectorize(class_to_index.get)(label_band)

print(unique_classes)
print(class_to_index)
print(index_to_class)
print(index_to_classname)



# Define the dataset
# Input stack is all the channels of the full stack except the last one
# Output stack is the indexed the land_cover

input_stack = standardized_stack[0:-1,...]
output_stack = indexed_label_band
dataset = LandCoverDataset(input_stack, output_stack.astype(np.float32), patch_size=16)
print(input_stack.shape)
print(output_stack.shape)

print(f"Input stack shape: {input_stack.shape}")
print(f"Output stack (classes stack) shape: {output_stack.shape}")

# Häufigkeiten normieren
frequencies = counts / np.sum(counts)

# Gewichtung berechnen (log stabilisiert)
weights = 1.0 / (np.log(1.02 + frequencies))
weights = weights / weights.sum()  # optional: Normierung auf Summe = 1
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

print("Gewichte:", weights_tensor)

# Check dataset length (number of patches of patch_size * patch_size)
len(dataset)

sample = dataset[0]

print("Image shape:", sample['image'].shape)
print("Label shape:", sample['label'].shape)


# Set the ratio for the train/test split
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Ensure a random and reproducible split by setting a manual seed
generator = torch.Generator().manual_seed(13)

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
# Note the parameter shuffle=True for the Training dataloader --> Each epoch the
#   training patches will be loaded in a different order
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle = False)

visualize_patch_split(stack_full, train_dataset, test_dataset, patch_size=16)
model = SimpleCNN(in_channels=input_stack.shape[0], num_classes=len(unique_classes)).to(device)  # Set num_classes appropriately
#model = UNet(input_stack.shape[0], len(unique_classes)).to(device)  # Set num_classes appropriately
print(len(unique_classes))
print("Unique classes:", unique_classes)
#model = UNet(input_stack.shape[0],len(unique_classes)).to(device)
print(model)


criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 51

train_losses = []
test_losses = []
test_accuracies = []
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs = batch['image'].to(device)     # (B, Nb, 32, 32)
        labels = batch['label'].to(device)     # (B, 32, 32)

        optimizer.zero_grad()
        outputs = model(inputs)                # (B, num_classes, 32, 32)

        loss = criterion(outputs, labels)      # CrossEntropy expects shape (B, C, H, W) vs (B, H, W)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluation loop (basic accuracy)
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    running_loss = 0.0
    best_accuracy = 0.0 

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)  # (B, num_classes, H, W)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  # (B, H, W)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

        avg_loss = running_loss / len(test_loader)
        test_losses.append(avg_loss)

    accuracy = correct_pixels / total_pixels
    test_accuracies.append(accuracy)
    if (epoch % 10) == 0:
      print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Test Pixel Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        current_best_model = model.state_dict().copy()
        

# Optional: Save the trained model
torch.save(current_best_model, f"../models/land_cover_cnn_whole_simple_correct{best_accuracy:.2f}.pth")


plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(np.asarray(test_accuracies)*100, label="test")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.tight_layout()

# Set model to eval mode
model.eval()

# Storage for all predictions and labels
all_preds = []
all_labels = []

all_dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

with torch.no_grad():
    for batch in all_dataloader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)  # (B, H, W)

        # Flatten and move to CPU
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

class_names = [index_to_classname[i] for i in sorted(index_to_classname.keys())]

# 1. Classification Report
print("Classification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=3,
    zero_division=0
))

# 2. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(index_to_classname.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="viridis", xticks_rotation=70)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

stack_full = dataset_data[training_bands_idx, : ,:]

# Ensure full_stack is standardized with the same mean and std computed for the training
mean = np.nanmean(stack_full[:-1], axis=(1, 2))  # exclude label layer
std = np.nanstd(stack_full[:-1], axis=(1, 2))

standardized_stack = np.copy(stack_full)
for b in range(stack_full.shape[0] - 1):
    standardized_stack[b] = (stack_full[b] - mean[b]) / std[b]

prediction_map = predict_full_image(model, standardized_stack, patch_size=32, device=device, mean=mean, std=std)

# Define a default value for unknown classes
default_class_value = np.nan  # or any other fallback like 0.0

# Vectorized mapping function with default
vectorized_map = np.vectorize(lambda x: index_to_class.get(x, default_class_value))

# Apply it to the predicted class indices
pred_class_val = vectorized_map(prediction_map)
#np.vectorize(index_to_class.get)(prediction_map)

plot_landcover(stack_full[-1], title="Original landcover")
plot_landcover(pred_class_val, title="Predicted landcover")