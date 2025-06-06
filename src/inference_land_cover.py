'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-06-06
Purpose : inference land cover from Sentinel-1 and Sentinel-2 data using a pre-trained model
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

class_to_index = {
    0.0: 0,
    10.0: 1,
    20.0: 2,
    30.0: 3,
    40.0: 4,
    50.0: 5,
    60.0: 6,
    80.0: 7,
    -1.0: 8   # Dummy-Wert für "Unknown/Other"
}
index_to_class = {v: k for k, v in class_to_index.items()}
index_to_classname = {
    0: 'Not Defined',
    1: 'Tree cover',
    2: 'Shrubland',
    3: 'Grassland',
    4: 'Cropland',
    5: 'Built-up',
    6: 'Bare/sparse vegetation',
    7: 'Permanent water bodies',
    8: 'Unknown/Other'
}
# Load the dataset
dataset_name = "../Palma_datastack_final.tif"
fullpath = dataset_name

with rasterio.open(fullpath) as dataset:
    dataset_data = dataset.read()
    bands = dataset.descriptions
    

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

# Display the rgb bands of the image after
rgb = np.dstack((dataset_data[17,:,:], dataset_data[18,:,:], dataset_data[19,:,:]))
rgb = rgb / np.nanpercentile(rgb, 98)
rgb = np.clip(rgb, 0, 1)
plot_data(rgb)

# for inference i will use the data auqired after the event
# Band Selection for training

training_bands = [
    'Amplitude_VV_20220108',
    'Amplitude_VH_20220108',
    'VH_VV_rate_20220108',
    'Sigma_Nought_VH_20220108',
    'RVI_20220108',
    'RWI_20220108',
    'MPDI_20220108',
    'S2_Red_20220103',
    'S2_Green_20220103',
    'S2_Blue_20220103',
    'NDVI_20220103',
    'NDWI_20220103',
    'AWEI_20220103',
    'NDBI_20220103',
    'NBR_20220103',
    'NDSI_20220103',
]


training_bands_idx = []
for band_training in training_bands:
    training_bands_idx.append(bands.index(band_training) if band_training in bands else -1)

stack_full = dataset_data[training_bands_idx, : ,:]

print(stack_full.shape)

print(np.mean(stack_full, axis=(1,2)))
print(np.std(stack_full, axis=(1,2)))



#for normalization
mean = np.nanmean(stack_full, axis=(1, 2))  
std = np.nanstd(stack_full, axis=(1, 2))

standardized_stack = np.copy(stack_full)
for b in range(stack_full.shape[0]):
    standardized_stack[b] = (stack_full[b] - mean[b]) / std[b]


print("standardized_stack shape:", standardized_stack.shape)

missing_bands = [b for b in training_bands if b not in bands]
if missing_bands:
    print("Warning: Missing bands:", missing_bands)

# Modell laden
#model = SimpleCNN(in_channels=standardized_stack.shape[0], num_classes=len(index_to_class)).to(device)
model = UNet(standardized_stack.shape[0], len(index_to_class)).to(device)

model.load_state_dict(torch.load("/home/danielbugelnig/AAU/6.Semester/AI4EO/project/models/land_cover_unet_simple_correct0.73.pth", map_location=device))
model.eval()

# Vorhersage
prediction_map = predict_full_image(model, standardized_stack, patch_size=32, device=device, mean=mean, std=std, inference=True)

# Mapping & Plot
vectorized_map = np.vectorize(lambda x: index_to_class.get(x, np.nan))
pred_class_val = vectorized_map(prediction_map)

plot_landcover(dataset_data[band_index], title="Original landcover")

plot_landcover(pred_class_val, title="Predicted landcover")