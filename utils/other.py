import dgl
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn

# Normalization layer for scaling data
class NormalizationLayer(nn.Module):
    
    def __init__(self, mean, std):
        
        super(NormalizationLayer, self).__init__()
        self.mean = mean  # Mean for normalization
        self.std = std    # Standard deviation for normalization

    # Normalize the input tensor
    def normalize(self, x):
        return (x - self.mean) / self.std

    # Denormalize the input tensor
    def denormalize(self, x):
        return x * self.std + self.mean


# Function to calculate MAE loss
def masked_mae_loss(y_pred, y_true):
    
    # Create a mask to ignore zero values in y_true
    mask = (y_true != 0).float()
    
    # Normalize the mask by dividing by its mean to avoid imbalance
    mask /= mask.mean()

    # Compute the absolute difference
    loss = torch.abs(y_pred - y_true)
    
    # Apply the mask to the loss
    loss = loss * mask

    # Set NaN values to 0
    loss[loss != loss] = 0

    # Return the mean of the masked loss
    return loss.mean()


# Function to get the current learning rate from an optimizer
def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param["lr"]
