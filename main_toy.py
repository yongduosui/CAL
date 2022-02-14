from train import train_baseline_toy
from train_causal import train_causal_toy
from opts import setup_seed
import torch
import opts
import utils
import pdb
import time
import warnings
warnings.filterwarnings('ignore')

def main():

    args = opts.parse_args()
    setup_seed(args.seed)
    class_list = ["house", "cycle", "grid", "diamond"]
    class_list = class_list[:args.num_classes]
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}
    
    save_path = "./dataset-class-{}-num{}-shape{}-node-{}-dim{}-noise{}.pt".format(args.num_classes,
                    args.data_num, 
                    args.shape_num, 
                    args.node_num,
                    args.max_degree,
                    args.noise)
    try:
        dataset = torch.load(save_path)

    except:
        dataset = utils.graph_dataset_generate(args, class_list, settings_dict, save_path)
    
    args.bias = [args.bias, 1 - args.bias, 1 - args.bias, 1 - args.bias]
    bias_list = args.bias
    bias_dict = {"house": bias_list[0], 
                 "cycle": bias_list[1],
                 "grid": bias_list[2],
                 "diamond": bias_list[3]}

    
    train_set, val_set, test_set, the = utils.dataset_bias_split(dataset, args, class_list, bias_dict=bias_dict, split=712, total=args.data_num * 4)
    group_counts = utils.print_dataset_info(train_set, val_set, test_set, class_list, the)
    args.the = the
    
    if args.model in ["GIN","GCN", "GAT", "TopK", "SAGPool", "DiffPool", "SortPool", "GATv2", "GlobalAttention", "AGNN", "SuperGAT", "SMG"]:
        model_func = opts.get_model(args)
        train_baseline_toy(train_set, val_set, test_set, model_func=model_func, args=args, group_counts=group_counts)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        model_func = opts.get_model(args)
        train_causal_toy(train_set, val_set, test_set, model_func=model_func, args=args, group_counts=group_counts)
    elif args.model in ["GK", "WL", "SP", "RW"]:
        train_kernel(train_set, val_set, test_set, args)
    else:
        assert False

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print("total time:{:.2f}".format((t1 - t0)))
    