import torch
from functools import partial
import dgl
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# Import utils
from utils.other import get_learning_rate
from utils.other import masked_mae_loss
from utils.other import NormalizationLayer

# Function for evaluating the model
def eval(model, graph, dataloader, normalizer, loss_fn, device, args):
    total_loss = []               # List to store the loss values for each batch
    graph = graph.to(device)      # Move the graph to the specified device (GPU/CPU)
    model.eval()                  # Set the model to evaluation mode
    batch_size = args.batch_size  # Get the batch size from args

    for i, (x, y) in enumerate(dataloader):

        # Adjust the batch size if it's smaller than the specified batch_size
        if x.shape[0] != batch_size:

            # Create buffer tensors for x and y to pad to the batch size
            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            
            # Copy existing data into the buffer
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :, :] = x[-1].repeat(batch_size - x.shape[0], 1, 1, 1)  # Repeat last element to fill
            
            y_buff[: y.shape[0], :, :, :] = y
            y_buff[y.shape[0]:, :, :, :] = y[-1].repeat(batch_size - y.shape[0], 1, 1, 1)  # Repeat last element to fill
            
            x = x_buff
            y = y_buff

        # Permute dimensions to match the expected shape for the model (time, batch, ...)
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        # Normalize the input and target data, and reshape for model input
        x_norm = (normalizer.normalize(x).reshape(x.shape[0], -1, x.shape[3]).float().to(device))
        y_norm = (normalizer.normalize(y).reshape(x.shape[0], -1, x.shape[3]).float().to(device))
        
        # Reshape y for loss computation
        y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)

        # Batch the graph to match the batch size
        batch_graph = dgl.batch([graph] * batch_size)

        # Pass the input data through the model
        output = model(batch_graph, x_norm, y_norm, i, device)

        # Denormalize the predicted output
        y_pred = normalizer.denormalize(output)

        # Calculate the loss between predictions and ground truth
        loss = loss_fn(y_pred, y)

        # Append the current batch's loss to the total loss list
        total_loss.append(float(loss))

    # Return the mean loss over all batches
    return np.mean(total_loss)
