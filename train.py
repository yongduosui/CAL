import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import tensor
from utils import k_fold
import torch_geometric.transforms as T
import pdb
import random
import numpy as np
from torch.autograd import grad
from torch_geometric.data import Batch

def process_dataset(dataset):
    
    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
    transform = T.ToDense(num_nodes)
    new_dataset = []
    
    for data in tqdm(dataset):
        data = transform(data)
        add_zeros = num_nodes - data.feat.shape[0]
        if add_zeros:
            dim = data.feat.shape[1]
            data.feat = torch.cat((data.feat, torch.zeros(add_zeros, dim)), dim=0)
        new_dataset.append(data)
    return new_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_baseline_toy(train_set, val_set, test_set, model_func=None, args=None, ckpt=None, group_counts=None):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.model == "DiffPool":
        
        save_path = "diff_data-bias-{}.pt".format(args.bias[0])
        try:
            dataset = torch.load(save_path)
            train_set, val_set, test_set = dataset['train_set'], dataset['val_set'], dataset['test_set']
        except:
            dataset = {}
            train_set = process_dataset(train_set)
            val_set = process_dataset(val_set)
            test_set = process_dataset(test_set)
            dataset['train_set'], dataset['val_set'], dataset['test_set'] = train_set, val_set, test_set
            torch.save(dataset, save_path)
        
        train_loader = DenseLoader(train_set, args.batch_size, shuffle=True)
        val_loader = DenseLoader(val_set, args.batch_size, shuffle=False)
        test_loader = DenseLoader(test_set, args.batch_size, shuffle=False)
    else:
        
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    model = model_func(args.feature_dim, args.num_classes).to(device)

    if args.inference:
        ckpt = torch.load("{}-bias{}.pt".format(args.model, args.bias[0]))
        model.load_state_dict(ckpt)
        eval_acc(model, test_loader, device, args, vis=True)
        return

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc, update_test_acc, update_train_acc, update_epoch = 0, 0, 0, 0
    
    if args.train_type == 'dro':
        adjustments = np.array([0.0] * 8)
        train_loss_computer = LossComputer(
            F.nll_loss,
            is_robust=True,
            group_counts=group_counts['Train'],
            adj=adjustments,
            step_size=args.step_size,
            normalize_loss=False,
            btl=False,
            min_var_weight=0)

    for epoch in range(1, args.epochs + 1):
        
        if args.train_type == "irm":
            train_loss, train_acc = train_irm(model, optimizer, train_loader, device, args)
        elif args.train_type == 'dro':
            train_loss, train_acc = train_dro(model, optimizer, train_loader, device, args, train_loss_computer)
        else:
            train_loss, train_acc = train(model, optimizer, train_loader, device, args)

        val_acc = eval_acc(model, val_loader, device, args, vis=False)
        if epoch > 100:
            # test_acc, test_detail_dicts = eval_acc_detail(model, test_loader, device, args.the)
            # print("-" * 100)
            # for k, v in test_detail_dicts.items():
            #     print(k, v)
            # print("-" * 100)
            test_acc = eval_acc(model, test_loader, device, args, vis=False)
        else:
            test_acc = eval_acc(model, test_loader, device, args, vis=False)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_test_acc = test_acc
            update_train_acc = train_acc
            update_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), "{}-bias{}.pt".format(args.model, args.bias[0]))
     
        print("BIAS:[{:.2f},{:.2f},{:.2f},{:.2f}] | BS:[{}] Shape Num:[{}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Update Test:[{:.2f}] at Epoch:[{}]"
                .format(args.bias[0],args.bias[1],args.bias[2],args.bias[3],
                        args.batch_size,
                        args.shape_num,
                        args.model,
                        epoch, 
                        args.epochs,
                        train_loss, 
                        train_acc * 100, 
                        val_acc * 100,
                        test_acc * 100, 
                        best_val_acc * 100,
                        update_test_acc * 100, 
                        update_epoch))

    print("syd: BIAS:[{:.2f},{:.2f},{:.2f},{:.2f}] | BS:[{}] Shape Num:[{}] | Train acc:[{:.2f}] Val acc:[{:.2f}] Test acc:[{:.2f}] at epoch:[{}]"
        .format(args.bias[0],args.bias[1],args.bias[2],args.bias[3], 
                args.batch_size,
                args.shape_num,
                update_train_acc * 100,
                best_val_acc * 100, 
                update_test_acc * 100,
                update_epoch))

def train_baseline(dataset=None, model_func=None, args=None):

    train_accs, test_accs = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
        # if fold > 0:
        #     break
        best_test_acc, best_epoch = 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc = train(model, optimizer, train_loader, device, args)
            test_acc = eval_acc(model, test_loader, device, args)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
 
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("Baseline | dataset:[{}] Model:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.4f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}]"
                    .format(args.dataset,
                            args.model,
                            fold,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch))

        print("syd: Baseline fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}]"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch))
    
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    if args.epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(args.folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()

    print('sydall Final | Dataset:[{}] Model:[{}] | Test Acc: {:.2f}±{:.2f} | \nsydal: Selected epoch:{}| acc list:{}'
         .format(args.dataset,
                 args.model,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 selected_epoch[0],
                 test_acc.tolist()))

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        x = data.x if data.x is not None else data.feat
        return x.size(0)
        
def train(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    correct = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        if args.model == "SuperGAT":
            out, att_loss = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
            loss += 4.0 * att_loss
        else:
            out = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train_dro(model, optimizer, loader, device, args, loss_computer):
    
    model.train()
    total_loss = 0
    correct = 0
    
    mean_weight = 0
    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        
        loss, weights = loss_computer.loss(out, data)
        mean_weight += weights
        # loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    mean_weight = mean_weight / (it + 1)
    print(mean_weight)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def train_irm(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        
        optimizer.zero_grad()
        data_env1, data_env2 = group_by_context(data, args.the)
        data = data.to(device)
        data_env1 = data_env1.to(device)
        data_env2 = data_env2.to(device)
        # out = model(data)
        logits1, out1 = model(data_env1, train_type="irm")
        logits2, out2 = model(data_env2, train_type="irm")
        
        loss1 = F.nll_loss(out1, data_env1.y.view(-1))
        loss2 = F.nll_loss(out2, data_env2.y.view(-1))

        penalty1 = compute_penalty(logits1, data_env1)
        penalty2 = compute_penalty(logits2, data_env2)
        loss = loss1 + loss2 + args.penalty_weight * (penalty1 + penalty2)

        pred1 = out1.max(1)[1]
        correct += pred1.eq(data_env1.y.view(-1)).sum().item()
        pred2 = out2.max(1)[1]
        correct += pred2.eq(data_env2.y.view(-1)).sum().item()

        # loss = F.nll_loss(out, data.y.view(-1))
        # pred = out.max(1)[1]
        # correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_acc(model, loader, device, args, vis=False):
    
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data, vis=vis)
            if vis:
                return
            if args.model == "SuperGAT":
                pred = output[0].max(1)[1]
            else:
                pred = output.max(1)[1]
            # if args.model == "SuperGAT":
            #     pred = model(data)[0].max(1)[1]
            # else:
            #     pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def eval_acc_detail(model, loader, device, thehold):

    class_list = ['house', 'cycle', 'grid', 'diamond']

    error_dicts = {}

    error_dicts['house'] = {}
    error_dicts['cycle'] = {}
    error_dicts['grid'] = {}
    error_dicts['diamond'] = {}

    error_dicts['house']['ba'] = [0,0,0,0]
    error_dicts['cycle']['ba'] = [0,0,0,0]
    error_dicts['grid']['ba'] = [0,0,0,0]
    error_dicts['diamond']['ba'] = [0,0,0,0]

    error_dicts['house']['tree'] = [0,0,0,0]
    error_dicts['cycle']['tree'] = [0,0,0,0]
    error_dicts['grid']['tree'] = [0,0,0,0]
    error_dicts['diamond']['tree'] = [0,0,0,0]

    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        data_list = data.to_data_list()
        with torch.no_grad():
            pred = model(data).max(1)[1]
            error_list = (pred!=data.y).nonzero().t().squeeze().tolist()
            if isinstance(error_list, int):
                error_list = [error_list]
            for idx in error_list:
                edge_num = data_list[idx].num_edges
                true_label = data_list[idx].y.item()
                false_label = pred[idx].item()
                if edge_num > thehold:
                    error_dicts[class_list[true_label]]['ba'][false_label] += 1
                else:
                    error_dicts[class_list[true_label]]['tree'][false_label] += 1
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset), error_dicts

def group_by_context(data, the):
    
    data_list = data.to_data_list()
    data_list1 = []
    data_list2 = []
    for data in data_list:
        if data.num_edges > the:
            data_list1.append(data)
        else:
            data_list2.append(data)
    data_env1, data_env2 = Batch.from_data_list(data_list1), Batch.from_data_list(data_list2)
    return data_env1, data_env2

def compute_penalty(logits, data_env):
    loss_function = torch.nn.CrossEntropyLoss()
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, data_env.y.view(-1))
    gradient = grad(loss, [scale], create_graph=True)[0]
    return torch.sum(gradient**2)