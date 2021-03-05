"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.loss import DiceLoss, CCELoss, ComboLoss
from utils.volume_stats import Dice3d, Jaccard3d
from networks.ResUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent
from torchvision import transforms


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a = 1e-2, nonlinearity='leaky_relu')  # He initialization
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
def data_stats(dataset, batch_size):
    transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            ])
    dataloader = DataLoader(SlicesDataset(dataset,transform), batch_size=batch_size, shuffle=False)
    mean_list = []
    std_list = []
   
    for i, item in enumerate(dataloader, 0):
        data = item['image'].squeeze()
        mean = torch.mean(data)
        std = torch.std(data)
        mean_list.append(mean)
        std_list.append(std)
        
    return torch.mean(torch.stack(mean_list)), torch.mean(torch.stack(std_list))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)            
            

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name
        self.batch_size = config.batch_size

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data transformation
        
        mean, std = data_stats(dataset, self.batch_size)
        transform_train = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0, hue=0),
                                    transforms.ToTensor(),
                                    AddGaussianNoise(0.01, 0.01),
                                    transforms.Normalize(mean=mean, std=std)
                                    ])
        
        transform_val = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                            ])
        
        # Create data loader

        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]],transform_train),
                batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]], transform_val),
                batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        self.test_data = dataset[split["test"]]

        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuring 2D version of modified UNet model in which plain conv units are replaced by
        # residual units
        
        self.model = UNet()
        
        # Weights are initialized with He initialization
        self.model.apply(weights_init)
        
        self.model.to(self.device)
        
        # The loss function consists of both categorical cross entropy (L_ce) and 
        # soft dice loss (L_dice) with 0 <= kappa <= 1 :
        # loss = kappa * L_ce + (1-kappa) * L_dice
        
        self.loss_function = ComboLoss()
    
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, threshold=1e-3)
        
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")
        
    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        print('Current learning rate: ', self.optimizer.param_groups[0]['lr'])
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # Feed data to the model and feed target to the loss function
            data = batch['image'].to(self.device, dtype=torch.float)
            target = batch['seg'].to(self.device)
            prediction = self.model(data)

            # We are also getting softmax'd version of prediction to output a probability map
            # so that we can see how the model converges to the solution
            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target)
            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete.")
                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)
            print(".", end='')
        print("\nTraining complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                
                data = batch['image'].to(self.device, dtype=torch.float)
                target = batch['seg'].to(self.device)
                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)

                loss = self.loss_function(prediction, target)

                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100)
        print('Validation complete')

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # Dice and Jaccard similarity coefficients for accuracy assessment
            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc
                })
            print(f"{x['filename']} Dice {dc:.4f} , Jaccard {jc: .4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list)}
        print(f"Mean dice {np.mean(dc_list):.4f} , Mean Jaccard {np.mean(jc_list): .4f}")
        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
