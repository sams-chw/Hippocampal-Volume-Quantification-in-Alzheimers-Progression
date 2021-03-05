import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def soft_dice(self, y_pred, y_true):
        target = y_true[:, 0, :]
        one_hot = torch.nn.functional.one_hot(target, num_classes=3).permute(0,3,1,2)
        pred = F.softmax(y_pred, dim=1)
        
        axes = tuple(range(2, len(y_pred.shape)))
        num = 2. * torch.sum(pred * one_hot , dim=axes)
        pred_sqr_sum = torch.sum(torch.pow(pred, 1), dim=axes)
        one_hot_sqr_sum = torch.sum(torch.pow(one_hot, 1), dim=axes)
        den = pred_sqr_sum + one_hot_sqr_sum
        
        dice_coef = ( num + self.smooth ) / ( den + self.smooth )
        dice_loss = 1. - torch.mean(dice_coef)

        return dice_loss

    def forward(self, y_pred, y_true):
        return self.soft_dice(y_pred, y_true)    


class CCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true[:, 0, :, :])   
    
    
class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.cce = CCELoss()
        self.gamma = 0.8

    def forward(self, y_pred, y_true):
        dice_loss = self.dice(y_pred, y_true)
        cce_loss = self.cce(y_pred, y_true)
        combo_loss = self.gamma*cce_loss + (1-self.gamma)*dice_loss
        return combo_loss
