'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-05-19
Purpose : Change detection of the vulcano eruption in La Palma
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



# Validation Metrics sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utility import plot_landcover, plot_data, class_mapping

dataset_name = "../data/Palma_datastack_change_detection.tif"
fullpath = dataset_name

with rasterio.open(fullpath) as dataset:
    dataset_data = dataset.read()
    bands = dataset.descriptions

band_name = "Change_Band"
# find the index of the desired band
band_index = bands.index(band_name)

band = dataset_data[band_index, :, :]
plot_data(band, band_name)