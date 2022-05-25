import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import tensor
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import pdb
import random
import numpy as np
from torch.autograd import grad
from torch_geometric.data import Batch
from utils import k_fold, num_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train_baseline_syn(train_set, val_set, test_set, model_func=None, args=None):
    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
    best_val_acc, update_test_acc, update_train_acc, update_epoch = 0, 0, 0, 0
    
    for epoch in range(1, args.epochs + 1):
        
        train_loss, train_acc = train(model, optimizer, train_loader, device, args)
        val_acc = eval_acc(model, val_loader, device, args)
        test_acc = eval_acc(model, test_loader, device, args)
        lr_scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_test_acc = test_acc
            update_train_acc = train_acc
            update_epoch = epoch
     
        print("BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Update Test:[{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                .format(args.bias,
                        args.model,
                        epoch, 
                        args.epochs,
                        train_loss, 
                        train_acc * 100, 
                        val_acc * 100,
                        test_acc * 100, 
                        best_val_acc * 100,
                        update_test_acc * 100, 
                        update_epoch,
                        optimizer.param_groups[0]['lr']))

    print("syd: BIAS:[{:.2f}] | Best Val acc:[{:.2f}] Test acc:[{:.2f}] at epoch:[{}]"
        .format(args.bias,
                best_val_acc * 100, 
                update_test_acc * 100,
                update_epoch))

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
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_acc(model, loader, device, args):
    
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)
