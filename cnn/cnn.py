import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import mad_std

from sklearn.metrics import precision_recall_curve

import configparser



class LAEDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                1,
                16,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.InstanceNorm3d(
                16,
                affine=True,
            ),
            nn.GELU(),
            nn.Conv3d(
                16,
                32,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            ),
            nn.InstanceNorm3d(
                32,
                affine=True,
            ),
            nn.GELU(),
            nn.AdaptiveMaxPool3d((None, 1, 1)),
        )
        self.spectral = nn.Sequential(

            nn.Conv1d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = self.spatial(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.spectral(x)
        x = self.head(x)
        return x


class LAEDetector3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.GELU(),
            
            nn.AdaptiveMaxPool3d((1, 1, 1))
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x