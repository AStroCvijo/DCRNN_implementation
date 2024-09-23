import torch
from functools import partial
import dgl
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the funcitons for data loading
from data.data_load import METR_LAGraphDataset
from data.data_load import METR_LATestDataset
from data.data_load import METR_LATrainDataset
from data.data_load import METR_LAValidDataset

# Import utils
from utils.argparser import arg_parse
from utils.other import get_learning_rate
from utils.other import masked_mae_loss
from utils.other import NormalizationLayer

# Import the model
from model.dcrnn import DiffConvLayer
from model.dcrnn import GraphRNN

# Import the functions for training and evaluation
from train.train import train
from train.eval import eval

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    if args.dataset == "LA":
        g = METR_LAGraphDataset()
        train_data = METR_LATrainDataset()
        test_data = METR_LATestDataset()
        valid_data = METR_LAValidDataset()
    
    # Initialize the data loaders
    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True
    )
    test_loader = DataLoader(
        test_data,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True
    )

    print("Data loaders initialized!")

    # Initialize the normalizer
    normalizer = NormalizationLayer(train_data.mean, train_data.std)

    # Initialize the model
    batch_g = dgl.batch([g] * args.batch_size).to(device)
    out_gs, in_gs = DiffConvLayer.attach_graph(batch_g, args.diffsteps)
    net = partial(DiffConvLayer, k=args.diffsteps, in_graph_list=in_gs, out_graph_list=out_gs)

    model = GraphRNN(in_feats=2, out_feats=64, seq_len=12, num_layers=2, net=net, decay_steps=args.decay_steps).to(device)

    # Initialize the optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    loss_fn = masked_mae_loss

    print("DCRNN model initialized!")
    print("Starting training...")

    # Initialize the batch count
    batch_cnt = [0]

    for e in range(args.epochs):

        # Call the train and evaluation functions and calculate the train, eval and test loss 
        train_loss = train(model, g, train_loader, optimizer, scheduler, normalizer, loss_fn, device, args, batch_cnt)
        valid_loss = eval(model, g, valid_loader, normalizer, loss_fn, device, args)
        test_loss = eval(model, g, test_loader, normalizer, loss_fn, device, args)

        print("\rEpoch: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(e, train_loss, valid_loss, test_loss))