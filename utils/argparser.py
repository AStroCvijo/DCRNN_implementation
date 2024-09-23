import argparse

# Function for parsing arguments
def arg_parse():

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument('-d',   '--dataset',          type=str,   default="LA",           help="Dataset that will be used")
    
    # Training arguments
    parser.add_argument('-t',   '--train',            action="store_true", default=False, help="Whether to train the model")
    parser.add_argument('-e',   '--epochs',           type=int,   default=100,            help="Number of epochs for training")
    parser.add_argument('-lr',  '--learning_rate',    type=float, default=0.01,           help="Initial learning rate")
    parser.add_argument('--minimum_lr',               type=float, default=2e-6,           help="Lower bound of learning rate")
    parser.add_argument('--batch_size',               type=int,   default=64,             help="Size of batch for minibatch training")
    parser.add_argument('--num_workers',              type=int,   default=0,              help="Number of workers for parallel dataloading")
    parser.add_argument('--model',                    type=str,   default="dcrnn",        help="Which model to use: DCRNN vs GaAN")
    parser.add_argument('--gpu',                      type=int,   default=-1,             help="GPU index -1 for CPU training")
    parser.add_argument('--diffsteps',                type=int,   default=2,              help="Step of constructing the diffusion matrix")
    parser.add_argument('--num_heads',                type=int,   default=2,              help="Number of multi-attention heads")
    parser.add_argument('--decay_steps',              type=int,   default=2000,           help="Teacher forcing probability decay ratio")
    parser.add_argument('--max_grad_norm',            type=float, default=5.0,            help="Maximum gradient norm for updating parameters")

    # Parse the arguments
    return parser.parse_args()
