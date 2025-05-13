'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-05-13
Purpose : Utility functions for data preparation and visualization.
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
from torch.utils.data import DataLoader, TensorDataset

# Validation Metrics sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches



def plot_data(data, title="", colorbar=False, **kwargs):
  """
  Function to visualise and plot image data.
  """
  plt.figure(figsize=(12,6))
  img = plt.imshow(data, **kwargs)
  plt.title(title)
  if colorbar:
      plt.colorbar(img)
  plt.tight_layout()
  plt.show()
  

LC_class_names = [
    'Tree cover',
    'Shrubland',
    'Grassland',
    'Cropland',
    'Built-up',
    'Bare / sparse vegetation',
    'Snow and ice',
    'Permanent water bodies',
    'Herbaceous wetland',
    'Mangroves',
    'Moss and lichen'
]

class_mapping = {
    0: "Not Defined",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

def plot_landcover(land_cover_data, ax=None, figsize=(12, 6), title='ESA WorldCover Land Cover',
                        legend_type='patches', legend_position='lower right', dpi=100):
    """
    Plot ESA WorldCover land cover data with class names instead of numerical codes.

    Parameters:
    -----------
    land_cover_data : numpy.ndarray
        2D array containing the ESA WorldCover land cover classification values
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Title for the plot
    legend_type : str, optional
        Type of legend to use: 'colorbar' or 'patches'
    legend_position : str, optional
        Position of the legend: 'right', 'bottom', 'left', or 'top'
    dpi : int, optional
        Resolution of the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """

    # ESA WorldCover standard colors (official colormap)
    esa_colors = {
        0: '#282828',   # Not Defined (dark gray)
        10: '#006400',  # Tree cover (dark green)
        20: '#FFBB22',  # Shrubland (orange)
        30: '#FFFF4C',  # Grassland (yellow)
        40: '#F096FF',  # Cropland (pink)
        50: '#FA0000',  # Built-up (red)
        60: '#B4B4B4',  # Bare/sparse vegetation (light gray)
        70: '#F0F0F0',  # Snow and ice (white)
        80: '#0064C8',  # Permanent water bodies (blue)
        90: '#0096A0',  # Herbaceous wetland (teal)
        95: '#00CF75',  # Mangroves (light green)
        100: '#FAE6A0'  # Moss and lichen (beige)
    }

    # Get unique classes in sorted order (excluding NaN)
    unique_classes = np.unique(land_cover_data)
    unique_classes = unique_classes[~np.isnan(unique_classes)]  # Remove NaN values
    unique_classes = np.sort(unique_classes)

    if ax is None:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        # Get the figure
        fig = ax.figure


    # Prepare masked data to handle NaN values
    masked_data = np.ma.masked_invalid(land_cover_data)

    if legend_type == 'colorbar':
        # Create class boundaries for the colormap
        bounds = np.concatenate([unique_classes - 2.5, [unique_classes[-1] + 2.5]])

        # Create color list for the classes present in the data
        colors = [esa_colors[int(cls)] for cls in unique_classes if int(cls) in esa_colors]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot the data
        im = ax.imshow(masked_data, cmap=cmap, norm=norm)

        # Create the colorbar with class names
        cbar = fig.colorbar(im, ax=ax, orientation='vertical' if legend_position in ['right', 'left'] else 'horizontal')
        cbar.set_ticks(unique_classes)
        cbar.ax.set_yticklabels([class_mapping[int(cls)] for cls in unique_classes]) if legend_position in ['right', 'left'] else \
        cbar.ax.set_xticklabels([class_mapping[int(cls)] for cls in unique_classes], rotation=45, ha='right')

    else:  # legend_type == 'patches'
        # Create color list for the classes present in the data
        colors = [esa_colors[int(cls)] for cls in unique_classes if int(cls) in esa_colors]
        cmap = ListedColormap(colors)

        # Create normalized colormap with values mapped directly to colors
        bounds = np.concatenate([unique_classes - 2.5, [unique_classes[-1] + 2.5]])
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot the data
        im = ax.imshow(masked_data, cmap=cmap, norm=norm)

        # Create legend patches
        legend_patches = [mpatches.Patch(color=esa_colors[int(cls)],
                                         label=f"{int(cls)}: {class_mapping[int(cls)]}")
                          for cls in unique_classes if int(cls) in esa_colors]

        # Add the legend
        ax.legend(handles=legend_patches, loc=legend_position, framealpha=0.7)
    # Set title and remove axes
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

