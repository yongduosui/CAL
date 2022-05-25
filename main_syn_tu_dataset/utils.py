import featgen
import gengraph
import random
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import StratifiedKFold
import pdb

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def k_fold(dataset, folds, epoch_select):
    
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))
    
    return train_indices, test_indices, val_indices

def creat_one_pyg_graph(context, shape, label, feature_dim, shape_num, settings_dict, args=None):
    if args is None:
        noise = 0
    else:
        noise = args.noise
    if feature_dim == -1:
        # use degree as feature
        feature = featgen.ConstFeatureGen(None, max_degree=args.max_degree)
    else:
        feature = featgen.ConstFeatureGen(np.random.uniform(0, 1, feature_dim))
    G, node_label = gengraph.generate_graph(basis_type=context,
                                            shape=shape,
                                            nb_shapes=shape_num,
                                            width_basis=settings_dict[context]["width_basis"],
                                            feature_generator=feature,
                                            m=settings_dict[context]["m"],
                                            random_edges=noise) 
    pyg_G = from_networkx(G)
    pyg_G.y = torch.tensor([label])
    return pyg_G, node_label

def graph_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}

    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("create shape:{}".format(shape))
        for i in tqdm(range(args.data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += "/syn_dataset.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def test_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": (args.node_num) ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}
    data_num = int(0.2 * args.data_num)
    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("test set create shape:{}".format(shape))
        for i in tqdm(range(data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += "/syn_dataset_test.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def dataset_bias_split(dataset, args, bias=None, split=None, total=20000):
    
    class_list = ["house", "cycle", "grid", "diamond"]
    bias_dict = {"house": bias, "cycle": 1 - bias, "grid": 1 - bias, "diamond": 1 - bias}
    
    ba_dataset = dataset['ba']
    tr_dataset = dataset['tree']
    
    train_split, val_split, test_split = float(split[0]) / 10, float(split[1]) / 10, float(split[2]) / 10
    assert train_split + val_split + test_split == 1
    train_num, val_num, test_num = total * train_split, total * val_split, total * test_split
    # blance class
    class_num = args.num_classes
    train_class_num, val_class_num, test_class_num = train_num / class_num, val_num / class_num, test_num / class_num
    train_list, val_list, test_list  = [], [], []
    edges_num = 0
    
    for shape in class_list:
        bias = bias_dict[shape]
        train_tr_num = int(train_class_num * bias)
        train_ba_num = int(train_class_num * (1 - bias))
        val_tr_num = int(val_class_num * bias)
        val_ba_num = int(val_class_num * (1 - bias))
        test_tr_num = int(test_class_num * 0.5)
        test_ba_num = int(test_class_num * 0.5)
        train_list += tr_dataset[shape][:train_tr_num] + ba_dataset[shape][:train_ba_num]
        val_list += tr_dataset[shape][train_tr_num:train_tr_num + val_tr_num] + ba_dataset[shape][train_ba_num:train_ba_num + val_ba_num]
        test_list += tr_dataset[shape][train_tr_num + val_tr_num:train_tr_num + val_tr_num + test_tr_num] + ba_dataset[shape][train_ba_num + val_ba_num:train_ba_num + val_ba_num + test_ba_num]
        _, e1 = print_graph_info(tr_dataset[shape][0], "Tree", shape)
        _, e2 = print_graph_info(ba_dataset[shape][0], "BA", shape)
        
        edges_num += e1 + e2
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    the = float(edges_num) / (class_num * 2)
    return train_list, val_list, test_list, the

def print_graph_info(G, c, o):
    print('-' * 100)
    print("| graph: {}-{} | nodes num:{} | edges num:{} |".format(c, o, G.num_nodes, G.num_edges))
    print('-' * 100)
    return G.num_nodes, G.num_edges

def print_dataset_info(train_set, val_set, test_set, the):
    class_list = ["house", "cycle", "grid", "diamond"]
    dataset_group_dict = {}
    dataset_group_dict["Train"] = dataset_context_object_info(train_set, "Train", class_list, the)
    dataset_group_dict["Val"] = dataset_context_object_info(val_set, "Val   ", class_list, the)
    dataset_group_dict["Test"] = dataset_context_object_info(test_set, "Test  ", class_list, the)
    return dataset_group_dict

def dataset_context_object_info(dataset, title, class_list, the):

    class_num = len(class_list)
    tr_list = [0] * class_num
    ba_list = [0] * class_num
    for g in dataset:
        if g.num_edges > the: # ba
            ba_list[g.y.item()] += 1
        else: # tree
            tr_list[g.y.item()] += 1
    total = sum(tr_list) + sum(ba_list)
    info = "{} Total:{}\n| Tree: House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BA  : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| All : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BIAS: House:{:.1f}%, Cycle:{:.1f}%, Grids:{:.1f}%, Diams:{:.1f}%"
    print("-" * 150)
    print(info.format(title, total, tr_list[0], tr_list[1], tr_list[2], tr_list[3],
                                    ba_list[0], ba_list[1], ba_list[2], ba_list[3],
                                    tr_list[0] +  ba_list[0],    
                                    tr_list[1] +  ba_list[1], 
                                    tr_list[2] +  ba_list[2], 
                                    tr_list[3] +  ba_list[3],
                                    100 *float(tr_list[0]) / (tr_list[0] +  ba_list[0]),
                                    100 *float(tr_list[1]) / (tr_list[1] +  ba_list[1]),
                                    100 *float(tr_list[2]) / (tr_list[2] +  ba_list[2]),
                                    100 *float(tr_list[3]) / (tr_list[3] +  ba_list[3]),
                     ))
    print("-" * 150)
    total_list = ba_list + tr_list
    group_counts = torch.tensor(total_list).float()
    return group_counts