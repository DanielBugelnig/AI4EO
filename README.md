
# AI4EO: Remote Sensing Analysis of a Natural Disaster

This repository contains all code, data references, and documentation for analysis of a natural disaster.


The goal is to investigate land and surface changes caused by a natural disaster using:

- **Remote sensing data** (Sentinel-1 SAR, Sentinel-2 multispectral, Land Cover maps)
- **Physical and mathematical indices** (NDVI, NDWI, NDBI, AWEI, NBR, NDSI, etc.)
- **Artificial intelligence tools** (for classification, change detection)

We selected the case study:

> **La Palma Volcano Eruption (Canary Islands, 2021)**  

## Folder Structure

```
project_root/
│
├── data/                      # All data used or generated
│
├── src/                       # Source code for all processing and analysis steps
│
├── doc/                       # Reports and project documentation
|Datastack

```


This repository includes:

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
