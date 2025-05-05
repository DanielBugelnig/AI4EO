
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

- ✅ **Dataset** – All required satellite data, preprocessed and organized (see `data/` and `dataset_link.txt`)
- ✅ **Codebase** – Scripts and a main Colab notebook for end-to-end processing and analysis
- ✅ **Report** – A concise summary of our methodology, results, and conclusions (`doc/report.pdf`)

All three components are packaged in a single `.zip` archive for submission via **Atenea**.

## Project Timeline

- **Deadline:** Second week of June (exact date on Atenea)
- **Optional Pre-evaluation:** Early dataset submission for instructor review

## Required Dataset Bands

Each group must include the following data:

- **2 Sentinel-1 GRD** acquisitions
- **2 Sentinel-2 L2A** acquisitions (<5% cloud cover)
- **1 Land Cover** layer

Our dataset satisfies these requirements and includes additional bands and metadata to improve analysis.

## Usage

1. Clone the repository
2. Open `AI4EO_Project.ipynb` in [Google Colab](https://colab.research.google.com/)
3. Follow the cells step by step to:
   - Load and visualize data
   - Compute indices
   - Run classification and change detection
   - Export results and visualizations

---

**2025 – AI4EO Project – Remote Sensing and Earth Observation, Mirea Lopez, Daniel Bugelnig**
