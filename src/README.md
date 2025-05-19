# 📁 Source files overview

## 1. `utility.py`
This file contains utility functions for data preparation and visualization:

- **`plot_data`**: Visualizes 2D image data with optional colorbars.
- **Land Cover Class Names and Mapping**: Maps numerical land cover codes to class names.
- **`plot_landcover`**: Visualizes land cover data using ESA WorldCover standard colors, with legend or colorbar.

**Purpose**: General-purpose utilities for plotting and handling land cover data.

---

## 2. `indices.py`
Defines classes for computing various indices from radar and optical data:

- **`RadarIndices`**: 
  - VH/VV ratio  
  - Radar Vegetation Index (RVI)  
  - Radar Water Index (RWI)  
  - Modified Polarization Difference Index (MPDI)

- **`OpticalIndices`**:  
  - NDVI, NDWI, AWEI, NDBI, NDSI, NBR

- **`EOIndices`**: Combines radar and optical indices into one interface.

**Purpose**: Provides functions for calculating physical and mathematical indices from remote sensing data.

---

## 3. `indices_calculation.py`
Calculates various indices for Sentinel-1 and Sentinel-2 datasets:

- Loads and reprojects Sentinel-1 & Sentinel-2 data to match land cover data.
- Computes radar and optical indices.
- Generates valid pixel masks and combines them.
- Creates RGB visualizations and saves as PNG.
- Stacks indices and metadata into a single GeoTIFF.

**Purpose**: Prepares remote sensing data by calculating indices and generating a unified data stack.

---

## 4. `data_preparation.py`
Prepares and explores the datasets:

- Loads Sentinel-1, Sentinel-2, and land cover data.
- Visualizes the data (RGB and false-color images).
- Ensures spatial alignment of datasets.

**Purpose**: Exploratory preprocessing to understand and align the datasets.

---

## 5. `change_detection.py`
Detects changes due to the La Palma volcano eruption:

- Loads the preprocessed data stack (`Palma_datastack_change_detection.tif`).
- Extracts and visualizes the `"Change_Band"` layer.

**Purpose**: Performs change detection analysis using preprocessed data. --> needs to be done

---

## 6. `convolutionalNN.py`
Defines a CNN for land cover classification:

- **`LandCoverDataset`**: Custom PyTorch dataset for extracting data patches.
- **`SimpleCNN`**: CNN model for pixel-wise classification.

**Utility Functions**:
- `visualize_patch_split`: Shows train/test patch split.
- `predict_full_image`: Predicts entire image using the trained CNN.

**Purpose**: Implements a CNN for land cover classification.

---

## 7. `land_cover_classification.py`
Trains a CNN for land cover classification:

- Loads preprocessed data stack and selects bands.
- Normalizes data and splits it into training/testing sets.
- Trains the CNN and evaluates performance.
- Visualizes results and generates a confusion matrix.

**Purpose**: Trains and evaluates a CNN model for land cover classification.

---

## 🧾 Summary

The `src/` folder includes scripts for:

- **Utility Functions**:  
  `utility.py` – Visualization and land cover handling

- **Index Calculations**:  
  `indices.py` – Radar & optical index functions

- **Data Preparation**:  
  `data_preparation.py`, `indices_calculation.py` – Dataset alignment and preprocessing

- **Change Detection**:  
  `change_detection.py` – Land cover change analysis

- **Land Cover Classification**:  
  `convolutionalNN.py`, `land_cover_classification.py` – CNN-based classification

**End-to-End Goal**: Enable complete processing, analysis, and visualization of remote sensing data for the La Palma volcano eruption case study.
