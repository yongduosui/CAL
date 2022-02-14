import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import tensor
from utils import k_fold
import pdb
import networkx as nx
import numpy as np
from torch.autograd import grad
from torch_geometric.data import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_causal_toy(train_set, val_set, test_set, model_func=None, args=None, group_counts=None):

    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    model = model_func(args.feature_dim, args.num_classes).to(device)
    
    if args.inference:
        ckpt = torch.load("{}-{}.pt".format(args.model, args.bias[0]))
        model.load_state_dict(ckpt)
        eval_acc_causal(model, test_loader, device, args, vis=True)
        return
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_acc, update_test_acc_co, update_test_acc_c, update_test_acc_o, update_epoch = 0, 0, 0, 0, 0

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
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_irm(model, optimizer, train_loader, device, args)
        elif args.train_type == 'dro':
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_dro(model, optimizer, train_loader, device, args, train_loss_computer)
        else:
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal(model, optimizer, train_loader, device, args)
        val_acc_co, val_acc_c, val_acc_o = eval_acc_causal(model, val_loader, device, args)
        # test_acc_co, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
        test_acc_co, test_acc_c, test_acc_o, detail_info = eval_acc_causal_detail(model, test_loader, device, args)
        # if epoch > 80:
        #     print("-" * 100)
        #     for k, v in detail_info.items():
        #         print(k, v)
        #     print("-" * 100)
        if val_acc_o > best_val_acc:
            best_val_acc = val_acc_o
            update_test_acc_co = test_acc_co
            update_test_acc_c = test_acc_c
            update_test_acc_o = test_acc_o
            update_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), "{}-{}.pt".format(args.model, args.bias[0]))
        
        print("BIAS:[{:.2f},{:.2f},{:.2f},{:.2f}] | Type:[{}] Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Update Test:[co:{:.2f},c:{:.2f},o:{:.2f}] at Epoch:[{}]"
                .format(args.bias[0],args.bias[1],args.bias[2],args.bias[3], args.train_type,
                        args.model,
                        epoch, 
                        args.epochs,
                        train_loss,
                        loss_c,
                        loss_o,
                        loss_co,
                        train_acc_o * 100, 
                        val_acc_o * 100,
                        test_acc_o * 100, 
                        update_test_acc_co * 100,
                        update_test_acc_c * 100,  
                        update_test_acc_o * 100, 
                        update_epoch))

    print("syd: BIAS:[{:.2f},{:.2f},{:.2f},{:.2f}] | Val acc:[{:.2f}] Test acc:[co:{:.2f},c:{:.2f},o:{:.2f}] at epoch:[{}]"
        .format(args.bias[0],args.bias[1],args.bias[2],args.bias[3],
                val_acc_o * 100,
                update_test_acc_co * 100,
                update_test_acc_c * 100,  
                update_test_acc_o * 100, 
                update_epoch))


def train_causal_baseline(dataset=None, model_func=None, args=None):

    train_accs, test_accs, test_accs_c, test_accs_o = [], [], [], []
    random_guess = 1.0 / dataset.num_classes
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
        # if fold > 0:
        #     break
        best_test_acc, best_epoch, best_test_acc_c, best_test_acc_o = 0, 0, 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, loss_c, loss_o, loss_co, train_acc = train_causal(model, optimizer, train_loader, device, args)
            test_acc, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_accs_c.append(test_acc_c)
            test_accs_o.append(test_acc_o)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_test_acc_c = test_acc_c
                best_test_acc_o = test_acc_o
            
            print("Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.4f}] Test:[{:.2f}] Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] | Test_o:[{:.2f}] Test_c:[{:.2f}]"
                    .format(args.dataset,
                            fold,
                            epoch, args.epochs,
                            train_loss, loss_c, loss_o, loss_co,
                            train_acc * 100,  
                            test_acc * 100, 
                            test_acc_o * 100,
                            test_acc_c * 100, 
                            random_guess*  100,
                            best_test_acc * 100, 
                            best_epoch,
                            best_test_acc_o * 100,
                            best_test_acc_c * 100))

        print("syd: Causal fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}] | Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f})"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch,
                        best_test_acc_o * 100,
                        best_test_acc_c * 100,
                        random_guess*  100))
    
    train_acc, test_acc, test_acc_c, test_acc_o = tensor(train_accs), tensor(test_accs), tensor(test_accs_c), tensor(test_accs_o)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    test_acc_c = test_acc_c.view(args.folds, args.epochs)
    test_acc_o = test_acc_o.view(args.folds, args.epochs)
    
    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(args.folds)
    
    _, selected_epoch2 = test_acc_o.mean(dim=0).max(dim=0)
    selected_epoch2 = selected_epoch2.repeat(args.folds)

    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_c = test_acc_c[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_o = test_acc_o[torch.arange(args.folds, dtype=torch.long), selected_epoch2]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    test_acc_c_mean = test_acc_c.mean().item()
    test_acc_c_std = test_acc_c.std().item()
    test_acc_o_mean = test_acc_o.mean().item()
    test_acc_o_std = test_acc_o.std().item()

    print("=" * 150)
    print('sydall Final: Causal | Dataset:[{}] Model:[{}] seed:[{}]| Test Acc: {:.2f}±{:.2f} | OTest: {:.2f}±{:.2f}, CTest: {:.2f}±{:.2f} (RG:{:.2f}) | [Settings] co:{},c:{},o:{},harf:{},dim:{},fc:{}'
         .format(args.dataset,
                 args.model,
                 args.seed,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 test_acc_o_mean * 100, 
                 test_acc_o_std * 100,
                 test_acc_c_mean * 100, 
                 test_acc_c_std * 100,
                 random_guess*  100,
                 args.co,
                 args.c,
                 args.o,
                 args.harf_hidden,
                 args.hidden,
                 args.fc_num))
    print("=" * 150)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
        

def train_causal_dro(model, optimizer, loader, device, args, loss_computer):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        one_hot_target = data.y.view(-1)
        
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random)
        co_loss2, weights = loss_computer.loss(o_logs, data)
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        
        loss = args.c * c_loss + args.o * o_loss + args.co * co_loss + args.penalty_weight * co_loss2
        
        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o


def train_causal_irm(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
 
        data_env1_index, data_env2_index = group_by_index(data, args.the)
        data = data.to(device)
      
        one_hot_target = data.y.view(-1)
        c_logs, o_belog, o_logs, co_logs = model(data, train_type="irm", eval_random=args.with_random)
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        penalty1 = compute_penalty(o_belog, data, data_env1_index)
        penalty2 = compute_penalty(o_belog, data, data_env2_index)

        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        
        loss = args.c * c_loss + args.o * o_loss + args.co * co_loss + args.penalty_weight * (penalty1 + penalty2)
        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o


def train_causal(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1)
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random)
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        loss = args.c * c_loss + args.o * o_loss + args.co * co_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def eval_acc_causal(model, loader, device, args, vis=False):
    
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random, vis=vis)
            if vis:
                return
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_causal_detail(model, loader, device, args):

    thehold = args.the
    eval_random = args.eval_random
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
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        data_list = data.to_data_list()

        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
            error_list = (pred_o!=data.y).nonzero().t().squeeze().tolist()
            
            if isinstance(error_list, int):
                error_list = [error_list]
            for idx in error_list:
                edge_num = data_list[idx].num_edges
                true_label = data_list[idx].y.item()
                false_label = pred_o[idx].item()
                if edge_num > thehold:
                    error_dicts[class_list[true_label]]['ba'][false_label] += 1
                else:
                    error_dicts[class_list[true_label]]['tree'][false_label] += 1

        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o, error_dicts


def group_by_index(data, the):
    
    data_list = data.to_data_list()
    data_list1 = []
    data_list2 = []
    for idx, data in enumerate(data_list):
        if data.num_edges > the:
            data_list1.append(idx)
        else:
            data_list2.append(idx)
    return data_list1, data_list2


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


def compute_penalty(logits, data, data_env_index):

    logits = logits[data_env_index]
    loss_function = torch.nn.CrossEntropyLoss()
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, data.y.view(-1)[data_env_index])
    gradient = grad(loss, [scale], create_graph=True)[0]
    return torch.sum(gradient**2)

# def compute_penalty(logits, data_env):
#     loss_function = torch.nn.CrossEntropyLoss()
#     scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
#     loss = loss_function(logits * scale, data_env.y.view(-1))
#     gradient = grad(loss, [scale], create_graph=True)[0]
#     return torch.sum(gradient**2)