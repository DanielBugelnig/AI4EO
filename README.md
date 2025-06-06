
# AI4EO Project: Remote Sensing Analysis of a Natural Disaster

This repository contains all code, data references, and documentation for our **AI4EO project**, a key part of the course evaluation. The project leverages satellite-based remote sensing data and artificial intelligence tools to study and analyze the environmental impact of a **real-world natural disaster**.

## Project Objective

The goal is to investigate land and surface changes caused by a natural disaster using:

- **Remote sensing data** (Sentinel-1 SAR, Sentinel-2 multispectral, Land Cover maps)
- **Physical and mathematical indices** (NDVI, NDWI, NDBI, AWEI, NBR, NDSI, etc.)
- **Artificial intelligence tools** (for classification, change detection, and visualization)

We selected the case study:

> **La Palma Volcano Eruption (Canary Islands, 2021)**  

## Folder Structure

```
project_root/
│
├── data/                      # All datasets used or generated
│   ├── raw/                   # Raw Sentinel-1, Sentinel-2, and land cover data
│   ├── processed/             # Aligned, cleaned, and ready-to-use datasets
│   ├── exported/              # GeoTIFFs, images, and visualization outputs
│   └── dataset_link.txt       # Link to dataset folder on Google Drive
│
├── src/                       # Source code for all processing and analysis steps
│   ├── preprocessing/         # Scripts for preprocessing and correction
│   ├── indices/               # Index calculations (NDVI, NDWI, etc.)
│   ├── classification/        # AI/ML models for classification and change detection
│   └── visualization/         # Map generation and plotting
│
├── doc/                       # Reports and project documentation
│   ├── report.pdf             # Final report with methodology and results
│   └── notes.md               # Additional documentation and notes
│
└── AI4EO_Project.ipynb        # Main notebook (Google Colab-compatible)
```

## Deliverables

As per the project requirements, this repository includes:

- **Dataset**:All required satellite data, preprocessed and organized (see `Palma_datastack_change_detection.tif` and `Palma_datastack_final.tif`)
- **Codebase**: This repository: `https://github.com/DanielBugelnig/AI4EO.git`
-  **Report**:(`report.pdf`)



## Usage

1. Clone the repository
2. Create a virtual environment (python3.10) install all requirements (requirements.txt)
3. Open `AI4EO_Project.ipynb` in [Google Colab](https://colab.research.google.com/)
4. Read the readme in the `/src` for explanations
---

**2025 – AI4EO Project – Remote Sensing and Earth Observation, Mirea Lopez, Daniel Bugelnig**
