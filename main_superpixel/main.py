import numpy as np
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from train.train_causal import train_epoch, evaluate_network
from nets.load_net import gnn_model
from data.data import LoadData
import opts
import copy
import pdb


def run_causal(dataset, net_params, args):

    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = torch.device("cuda")
    
    model = gnn_model(args.model, net_params, args)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=10,
                                                     verbose=True)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=False, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, collate_fn=dataset.collate)
    run_time, best_val_acc, best_epoch, update_test_acc_co  = 0, 0, 0, 0
    for epoch in range(1, args.epochs + 1):

        t0 = time.time()
        epoch_train_loss_dict, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, args)
        epoch_val_loss, epoch_val_acc_dict = evaluate_network(model, device, val_loader, epoch, args)
        _, epoch_test_acc_dict = evaluate_network(model, device, test_loader, epoch, args)                
        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc_dict['o'] > best_val_acc:
            best_val_acc = epoch_val_acc_dict['o']
            update_test_acc_co = epoch_test_acc_dict['co']
            update_test_acc_c = epoch_test_acc_dict['c']
            update_test_acc_o = epoch_test_acc_dict['o']
            best_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), "{}_mnist_causal-c{}-o-{}-co-{}.pt".format(args.model, args.c, args.o, args.co))

        print('-'*120)
        print('Causal:[{}] | Epoch [{}/{}]: Loss [{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[c:{:.2f},o:{:.2f},co:{:.2f}] | Update Val:[{:.2f}] Test:[c:{:.2f},o:{:.2f},co:{:.2f}] at epoch:[{}] | Run Total Time: [{:.2f} min]'
                .format(args.dataset,
                        epoch, 
                        args.epochs,
                        epoch_train_loss_dict['total'], 
                        epoch_train_loss_dict['c'],
                        epoch_train_loss_dict['o'],
                        epoch_train_loss_dict['co'],
                        epoch_train_acc * 100,
                        epoch_val_acc_dict['o'] * 100, 
                        epoch_test_acc_dict['c'] * 100,
                        epoch_test_acc_dict['o'] * 100,
                        epoch_test_acc_dict['co'] * 100,
                        best_val_acc * 100,
                        update_test_acc_c * 100,
                        update_test_acc_o * 100,
                        update_test_acc_co * 100,
                        best_epoch,
                        run_time / 60)) 
        print('-'*120)
        
    print("syd: Causal:[{}] | Update Val:[{:.2f}] Test:[c:{:.2f},o:{:.2f},co:{:.2f}] at epoch:[{}] | Settings:[c:{:.1f},o:{:.1f},co:{:.1f}]"
            .format(args.dataset,
                    best_val_acc * 100,
                    update_test_acc_c * 100,
                    update_test_acc_o * 100,
                    update_test_acc_co * 100,
                    best_epoch,
                    args.c, args.o, args.co))

def main():    

    args = opts.parser_loader() 
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME, args)

    params = config['params']
    params['seed'] = int(args.seed)
    net_params = config['net_params']
    net_params['L'] = args.layer
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes
    run_causal(dataset, net_params, args)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("totel:{:.2f} min".format((end_time - start_time) / 60))
