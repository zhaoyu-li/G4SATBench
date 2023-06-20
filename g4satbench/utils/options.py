import argparse


def add_model_options(parser):
    parser.add_argument('--graph', type=str, choices=['lcg', 'vcg'], default='lcg', help='Graph construction')
    parser.add_argument('--init_emb', type=str, choices=['learned', 'random'], default='learned', help='Embedding initialization')
    parser.add_argument('--model', type=str, choices=['neurosat', 'ggnn', 'ggnn*', 'gcn', 'gcn*', 'gin', 'gin*'], default='neurosat', help='GNN model')

    parser.add_argument('--dim', type=int, default=128, help='Dimension of embeddings and hidden states')
    parser.add_argument('--n_iterations', type=int, default=32, help='Number of iterations for message passing')
    
    parser.add_argument('--n_mlp_layers', type=int, default=2, help='Number of layers in all MLPs')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function in all MLPs')
