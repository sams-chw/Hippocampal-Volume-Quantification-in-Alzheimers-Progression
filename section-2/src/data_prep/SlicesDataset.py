"""
Module for Pytorch dataset representations
"""

import torch
import PIL
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data, transform=None):
        
        self.data = data
        self.slices = []
        self.transform = transform
        
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Total image files: ", len(data))
        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))
        print(f'Total image slices: {len(self.slices)}.')

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx
        
        if self.transform is not None:
            # apply image transformation
            img = self.data[slc[0]]['image'][slc[1]]
            img = np.uint8(img/np.max(img)*255)
            img = self.transform(img)
            sample['image'] = img
        else:
            sample['image'] = torch.from_numpy(self.data[slc[0]]['image'][slc[1]]).unsqueeze(0).to(self.device)

        sample['seg'] = torch.from_numpy(self.data[slc[0]]['seg'][slc[1]][None, :]).long().to(self.device)
        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
