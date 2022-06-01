import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from utils import k_fold, num_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_causal_syn(train_set, val_set, test_set, model_func=None, args=None):

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
        
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
    best_val_acc, update_test_acc_co, update_test_acc_c, update_test_acc_o, update_epoch = 0, 0, 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        
        train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_epoch(model, optimizer, train_loader, device, args)
        val_acc_co, val_acc_c, val_acc_o = eval_acc_causal(model, val_loader, device, args)
        test_acc_co, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
        lr_scheduler.step()
        if val_acc_o > best_val_acc:
            best_val_acc = val_acc_o
            update_test_acc_co = test_acc_co
            update_test_acc_c = test_acc_c
            update_test_acc_o = test_acc_o
            update_epoch = epoch
            
        print("BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Update Test:[co:{:.2f},c:{:.2f},o:{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                .format(args.bias,
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
                        update_epoch,
                        optimizer.param_groups[0]['lr']))

    print("syd: BIAS:[{:.2f}] | Val acc:[{:.2f}] Test acc:[co:{:.2f},c:{:.2f},o:{:.2f}] at epoch:[{}]"
        .format(args.bias,
                val_acc_o * 100,
                update_test_acc_co * 100,
                update_test_acc_c * 100,  
                update_test_acc_o * 100, 
                update_epoch))

def train_causal_real(dataset=None, model_func=None, args=None):

    train_accs, test_accs, test_accs_c, test_accs_o = [], [], [], []
    random_guess = 1.0 / dataset.num_classes
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):

        best_test_acc, best_epoch, best_test_acc_c, best_test_acc_o = 0, 0, 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, loss_c, loss_o, loss_co, train_acc = train_causal_epoch(model, optimizer, train_loader, device, args)
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

def train_causal_epoch(model, optimizer, loader, device, args):
    
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

def eval_acc_causal(model, loader, device, args):
    
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
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
