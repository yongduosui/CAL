import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import pdb
import os

device = torch.device('cuda')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()
    
def parser_loader():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    parser.add_argument('--without_random', type=str2bool, default=False)
    parser.add_argument('--see_num', type=int, default=5,  help="Please give a value for seed")
    parser.add_argument('--offset', type=int, default=8,  help="Please give a value for seed")
    parser.add_argument('--mode', type=str, default="train", help="Please give a value for model name")
    parser.add_argument('--seed', type=int, default=41,  help="Please give a value for seed")
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--epochs', type=int, default=100, help="Please give a value for epochs")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', type=str, help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--layer', type=int, default=4, help="Please give a value for epochs")
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--o', type=float, default=0.5)
    parser.add_argument('--co', type=float, default=0.5)
    parser.add_argument('--random_or_avg', type=str, help="random")
    parser.add_argument('--eval_random', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=False)

    args = parser.parse_args()
    print_args(args)
    setup_seed(args.seed)
    return args