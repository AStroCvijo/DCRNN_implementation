import os
import ssl
import dgl
import numpy as np
import torch
from six.moves import urllib
from torch.utils.data import DataLoader, Dataset

# Function for downloading the dataset
def download_file(dataset):

    print("Start Downloading data: {}".format(dataset))

    # Dataset URL
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/{}".format(dataset)

    print("Start Downloading File....")

    # Create an unverified SSL context to bypass SSL certificate verification
    context = ssl._create_unverified_context()

    # Download the file
    data = urllib.request.urlopen(url, context=context)

    # Write the downloaded data to a local file
    with open("./data/{}".format(dataset), "wb") as handle:
        handle.write(data.read())

# Custom dataset class for loading snapshots
class SnapShotDataset(Dataset):
    
    def __init__(self, path, npz_file):
        
        # Check if the dataset file exists
        if not os.path.exists(path + "/" + npz_file):
            if not os.path.exists(path):
                os.mkdir(path)
            download_file(npz_file)

        # Load the data from the NPZ file
        zipfile = np.load(path + "/" + npz_file)
        self.x = zipfile["x"]  # Load feature data
        self.y = zipfile["y"]  # Load target data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx, ...], self.y[idx, ...]

# Function to load the graph for the METR-LA dataset
def METR_LAGraphDataset():
    
    # Check if the graph file exists
    if not os.path.exists("data/graph_la.bin"):
        if not os.path.exists("data"):
            os.mkdir("data")
        download_file("graph_la.bin")

    # Load the graph using DGL
    g, _ = dgl.load_graphs("data/graph_la.bin")
    return g[0]  # Return the first graph

# Custom dataset class for the METR-LA training dataset
class METR_LATrainDataset(SnapShotDataset):
    
    def __init__(self):

        # Initialize using the training data NPZ file
        super(METR_LATrainDataset, self).__init__("data", "metr_la_train.npz")

        # Compute the mean and standard deviation for normalization
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()

# Custom dataset class for the METR-LA test dataset
class METR_LATestDataset(SnapShotDataset):
    
    def __init__(self):
        
        # Initialize using the test data NPZ file
        super(METR_LATestDataset, self).__init__("data", "metr_la_test.npz")

# Custom dataset class for the METR-LA validation dataset
class METR_LAValidDataset(SnapShotDataset):
    
    def __init__(self):

        # Initialize using the validation data NPZ file
        super(METR_LAValidDataset, self).__init__("data", "metr_la_valid.npz")
