import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import dgl.nn as dglnn
from dgl.base import DGLError

class DiffConvLayer(nn.Module):

    # Constructor method
    def __init__(self, in_feats, out_feats, k, in_graph_list, out_graph_list, dir="both"):
        super(DiffConvLayer, self).__init__()   
        self.in_feats = in_feats    # Input feature dimension
        self.out_feats = out_feats  # Output feature dimension
        self.k = k                  # Number of diffusion steps
        self.dir = dir              # Direction of diffusion

        # Number of graphs
        self.num_graphs = self.k - 1

        # Linear projections (one per diffusion step)
        self.project_fcs = nn.ModuleList()
        for _ in range(self.num_graphs):
            self.project_fcs.append(nn.Linear(self.in_feats, self.out_feats, bias=False))

        # Merger weights to combine features from different diffusion steps
        self.merger = nn.Parameter(torch.randn(self.num_graphs + 1))

        # Lists of precomputed inward and outward diffusion graphs
        self.in_graph_list = in_graph_list
        self.out_graph_list = out_graph_list

    @staticmethod
    def attach_graph(g, k):

        # Get the device
        device = g.device

        # Initialize out and in graph lists
        out_graph_list = []
        in_graph_list = []

        # Get weighted adjacency matrix, in-degrees, and out-degrees
        wadj, ind, outd = DiffConvLayer.get_weight_matrix(g)

        # Compute outward diffusion graphs
        adj = sparse.coo_matrix(wadj / outd.cpu().numpy())
        outg = dgl.from_scipy(adj, eweight_name="weight").to(device)
        outg.edata["weight"] = outg.edata["weight"].float().to(device)
        out_graph_list.append(outg)
        for _ in range(k - 1):
            out_graph_list.append(DiffConvLayer.diffuse(out_graph_list[-1], wadj, outd))

        # Compute inward diffusion graphs
        adj = sparse.coo_matrix(wadj.T / ind.cpu().numpy())
        ing = dgl.from_scipy(adj, eweight_name="weight").to(device)
        ing.edata["weight"] = ing.edata["weight"].float().to(device)
        in_graph_list.append(ing)
        for _ in range(k - 1):
            in_graph_list.append(DiffConvLayer.diffuse(in_graph_list[-1], wadj.T, ind))

        return out_graph_list, in_graph_list


    # Extract the weighted adjacency matrix and node degrees from the graph.
    @staticmethod
    def get_weight_matrix(g):

        adj = g.adj_external(scipy_fmt="coo")
        ind = g.in_degrees()
        outd = g.out_degrees()
        weight = g.edata["weight"]
        adj.data = weight.cpu().numpy()  # Assign edge weights to adjacency matrix
        return adj, ind, outd

    # Perform one step of diffusion on the current graph.
    @staticmethod
    def diffuse(progress_g, weighted_adj, degree):

        # Get the device
        device = progress_g.device

        progress_adj = progress_g.adj_external(scipy_fmt="coo")
        progress_adj.data = progress_g.edata["weight"].cpu().numpy()
        
        # Perform matrix multiplication for diffusion
        ret_adj = sparse.coo_matrix(progress_adj @ (weighted_adj / degree.cpu().numpy()))
        ret_graph = dgl.from_scipy(ret_adj, eweight_name="weight").to(device)
        ret_graph.edata["weight"] = ret_graph.edata["weight"].float().to(device)
        return ret_graph

    # Forward method
    def forward(self, g, x):

        # Initialize feature list to store features from each diffusion step
        feat_list = []

        # Use both inward and outward graphs
        graph_list = self.in_graph_list + self.out_graph_list

        # Perform diffusion convolution on each graph
        for i in range(self.num_graphs):

            g = graph_list[i]
            with g.local_scope():

                # Project node features to output dimension
                g.ndata["n"] = self.project_fcs[i](x)

                # Message passing: u_mul_e (multiply node features by edge weights) and sum
                g.update_all(fn.u_mul_e("n", "weight", "e"), fn.sum("e", "feat"))

                # Collect the updated node features after this diffusion step
                feat_list.append(g.ndata["feat"])

        # Final projection of the input node features
        feat_list.append(self.project_fcs[-1](x))

        # Combine features from all diffusion steps using the merger weights
        feat_list = torch.cat(feat_list).view(len(feat_list), -1, self.out_feats)
        ret = (self.merger * feat_list.permute(1, 2, 0)).permute(2, 0, 1).mean(0)

        return ret

# Define the GRUCell
class GraphGRUCell(nn.Module):

    # Constructor method
    def __init__(self, in_feats, out_feats, net):
        super(GraphGRUCell, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        # Three GNNs for reset, update, and candidate gates
        self.r_net = net(in_feats + out_feats, out_feats)
        self.u_net = net(in_feats + out_feats, out_feats)
        self.c_net = net(in_feats + out_feats, out_feats)

        # Learnable bias terms for each gate
        self.r_bias = nn.Parameter(torch.rand(out_feats))
        self.u_bias = nn.Parameter(torch.rand(out_feats))
        self.c_bias = nn.Parameter(torch.rand(out_feats))

    # Forward method
    def forward(self, g, x, h):
        
        # Reset gate (r): determines how much of previous hidden state to keep
        r = torch.sigmoid(self.r_net(g, torch.cat([x, h], dim=1)) + self.r_bias)
        
        # Update gate (u): how much of the current input contributes to the new hidden state
        u = torch.sigmoid(self.u_net(g, torch.cat([x, h], dim=1)) + self.u_bias)
        
        # Candidate hidden state (c): new candidate value to store in hidden state
        h_ = r * h
        c = torch.sigmoid(self.c_net(g, torch.cat([x, h_], dim=1)) + self.c_bias)
        
        # Final new hidden state: combination of update gate and candidate state
        new_h = u * h + (1 - u) * c
        return new_h

# Define the Stacked Encoder
class StackedEncoder(nn.Module):

    # Constructor method
    def __init__(self, in_feats, out_feats, num_layers, net):
        super(StackedEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.layers = nn.ModuleList()

        # First layer has input size in_feats, output size out_feats
        self.layers.append(GraphGRUCell(self.in_feats, self.out_feats, self.net))

        # Additional layers have input size out_feats, output size out_feats
        for _ in range(self.num_layers - 1):
            self.layers.append(GraphGRUCell(self.out_feats, self.out_feats, self.net))

    # Forward method
    def forward(self, g, x, hidden_states):

        hiddens = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x, hidden_states[i])
            hiddens.append(x)
        return x, hiddens

# Define the Stacked Decoder
class StackedDecoder(nn.Module):

    # Constructor method
    def __init__(self, in_feats, hid_feats, out_feats, num_layers, net):
        super(StackedDecoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net

        # Output layer for final prediction
        self.out_layer = nn.Linear(self.hid_feats, self.out_feats)
        self.layers = nn.ModuleList()

        # First layer has input size in_feats, output size hid_feats
        self.layers.append(GraphGRUCell(self.in_feats, self.hid_feats, net))

        # Additional layers have input size hid_feats, output size hid_feats
        for _ in range(self.num_layers - 1):
            self.layers.append(GraphGRUCell(self.hid_feats, self.hid_feats, net))

    # Forward method
    def forward(self, g, x, hidden_states):

        hiddens = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x, hidden_states[i])
            hiddens.append(x)
        # Apply linear transformation to the final layer's output
        x = self.out_layer(x)
        return x, hiddens

# Define the full DCRNN model that handles sequence to sequence predictions
class GraphRNN(nn.Module):

    # Constructor method
    def __init__(self, in_feats, out_feats, seq_len, num_layers, net, decay_steps):
        super(GraphRNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.net = net
        self.decay_steps = decay_steps

        # Encoder for input sequence
        self.encoder = StackedEncoder(self.in_feats, self.out_feats, self.num_layers, self.net)

        # Decoder for output sequence
        self.decoder = StackedDecoder(self.in_feats, self.out_feats, self.in_feats, self.num_layers, self.net)

    # Method for computing threshold for teacher forcing based on batch count
    def compute_thresh(self, batch_cnt):
        return self.decay_steps / (self.decay_steps + np.exp(batch_cnt / self.decay_steps))

    # Method for encoding the input sequence
    def encode(self, g, inputs, device):
        hidden_states = [torch.zeros(g.num_nodes(), self.out_feats).to(device) for _ in range(self.num_layers)]
        for i in range(self.seq_len):
            _, hidden_states = self.encoder(g, inputs[i], hidden_states)
        return hidden_states

    # Method for decoding the sequence, applying teacher forcing
    def decode(self, g, teacher_states, hidden_states, batch_cnt, device):
        outputs = []
        inputs = torch.zeros(g.num_nodes(), self.in_feats).to(device)
        for i in range(self.seq_len):
            if np.random.random() < self.compute_thresh(batch_cnt) and self.training:
                inputs, hidden_states = self.decoder(g, teacher_states[i], hidden_states)
            else:
                inputs, hidden_states = self.decoder(g, inputs, hidden_states)
            outputs.append(inputs)
        outputs = torch.stack(outputs)
        return outputs

    # Forward method
    def forward(self, g, inputs, teacher_states, batch_cnt, device):
        hidden = self.encode(g, inputs, device)
        outputs = self.decode(g, teacher_states, hidden, batch_cnt, device)
        return outputs