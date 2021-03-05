"""
Contains class that runs inferencing
"""
import torch
import sys
import numpy as np
from networks.ResUNet import UNet
from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet()

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))
            print('model weights loaded.')

        self.model.to(device)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # volume is a numpy array of shape [X,Y,Z] and I will slice X axis
        slices = []

        # create mask for each slice across the X (0th) dimension.
        # put all slices into a 3D Numpy array
        for ix in range(0, volume.shape[0]):
            slice_tensor = torch.from_numpy(volume[ix,:,:].astype(np.single)).unsqueeze(0).unsqueeze(0)
            pred = self.model(slice_tensor.to(self.device))
            y = pred.detach().squeeze(0)
            mask = torch.argmax(np.squeeze(pred.cpu().detach()), dim=0)
            slices.append(mask)
        return np.dstack(slices).transpose(2, 0, 1)

    def single_volume_inference_unpadded(self, volume, patch_size):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """

        volume = med_reshape(volume, (volume.shape[0], patch_size, patch_size))

        return self.single_volume_inference(volume)
