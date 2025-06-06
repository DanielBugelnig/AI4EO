'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-06-06
Purpose : script for architectural design and training of a convolutional neural network for land cover classification
'''



# PyTorch DL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt



from torch.utils.data import Dataset

def init_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        #Apply He for ReLU
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

class LandCoverDataset(Dataset):
    def __init__(self, input_stack, output_array, patch_size=32, transform=None):
        """
        Args:
            input_stack (ndarray): Numpy array of shape (Nb, H, W).
            output_array (ndarray): Numpy array of shape (H, W).
            patch_size (int): Size of square patches to extract.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patch_size = patch_size
        self.transform = transform

        # Separate input and labels
        self.inputs = input_stack  # (Nb, H, W)
        self.labels = output_array   # (H, W)

        # Extract dimensions
        _, self.H, self.W = self.inputs.shape

        # Compute number of patches
        self.patches = []
        for i in range(0, self.H - patch_size + 1, patch_size):
            for j in range(0, self.W - patch_size + 1, patch_size):
                input_patch = self.inputs[:, i:i+patch_size, j:j+patch_size]
                label_patch = self.labels[i:i+patch_size, j:j+patch_size]
                # Check for NaNs in the patch. If there are no NaNs add it
                if not (np.isnan(input_patch).any() or np.isnan(label_patch).any()):
                    self.patches.append((i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        i, j = self.patches[idx]

        # Extract patch
        input_patch = self.inputs[:, i:i+self.patch_size, j:j+self.patch_size]
        label_patch = self.labels[i:i+self.patch_size, j:j+self.patch_size]

        sample = {'image': torch.tensor(input_patch, dtype=torch.float32),
                  'label': torch.tensor(label_patch, dtype=torch.long)}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define a simple CNN model for pixel-wise classification
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)  # 1x1 conv for pixel-wise classification
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)  # Output shape: (B, num_classes, H, W)
        return x


def visualize_patch_split(full_stack, train_subset, test_subset, patch_size):
    H, W = full_stack.shape[1:]
    vis_mask = np.zeros((H, W), dtype=np.uint8)  # 0: ignored, 1: train, 2: test

    # Access the original dataset
    base_dataset = train_subset.dataset  # both train/test share the same base

    def mark_subset(subset, value):
        for idx in subset.indices:
            i, j = base_dataset.patches[idx]
            vis_mask[i:i+patch_size, j:j+patch_size] = value

    mark_subset(train_subset, value=1)  # 1 = train
    mark_subset(test_subset, value=2)   # 2 = test

    # Custom colormap: 0=blue (ignored), 1=green (train), 2=red (test)
    custom_cmap = ListedColormap(['blue', 'green', 'red'])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_mask, cmap=custom_cmap, interpolation='nearest')
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Ignored', 'Train', 'Test'])
    plt.title("Patch Sampling Map")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
def predict_full_image(model, full_stack, patch_size, device, mean, std, inference=False):
    model.eval()
    H, W = full_stack.shape[1:]
    if inference:
        endindex = 0
    else:       
        endindex = 1
    n_classes = model(torch.randn(1, full_stack.shape[0] - endindex, patch_size, patch_size).to(device)).shape[1]

    # Prepare output map and validity mask
    prediction_map = np.full((H, W), fill_value=np.nan)
    count_map = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            if inference:
                patch = full_stack[:, i:i+patch_size, j:j+patch_size]
            else:    
                patch = full_stack[:-1, i:i+patch_size, j:j+patch_size]

            if np.isnan(patch).any():
                continue  # skip patches with NaNs

            # Inference
            input_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).cpu().numpy()[0]

            # Fill prediction map
            prediction_map[i:i+patch_size, j:j+patch_size] = pred_class
            count_map[i:i+patch_size, j:j+patch_size] += 1

    return prediction_map