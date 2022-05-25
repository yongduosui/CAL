import torch
import torch.nn as nn
import time
from train.metrics import accuracy_MNIST_CIFAR as accuracy
import pdb

def train_epoch(model, optimizer, device, data_loader, epoch, args):

    model.train()
    epoch_loss = 0
    epoch_loss_c = 0
    epoch_loss_o = 0
    epoch_loss_co = 0
    epoch_train_acc_c = 0
    epoch_train_acc_o = 0
    epoch_train_acc_co = 0
    nb_data = 0
    gpu_mem = 0

    loss_dict = {}
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        xc, xo, xco = model.forward(batch_graphs, batch_x, batch_e, True)
        uniform_target = torch.ones(batch_labels.size(0), model.n_classes, dtype=torch.float).to(device) / model.n_classes
        
        loss_c = model.kl_loss(xc, uniform_target)
        loss_o = model.loss(xo, batch_labels)
        loss_co = model.loss(xco, batch_labels)
        loss = args.c * loss_c + args.o * loss_o + args.co * loss_co
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_loss_c += loss_c.detach().item()
        epoch_loss_o += loss_o.detach().item()
        epoch_loss_co += loss_co.detach().item()

        epoch_train_acc_c += accuracy(xc, batch_labels)
        epoch_train_acc_o += accuracy(xo, batch_labels)
        epoch_train_acc_co += accuracy(xco, batch_labels)
        nb_data += batch_labels.size(0)
    
        if iter % 100 == 0:
            print('-'*120)
            print('Epoch: [{}/{}]  Iter: [{}/{}]  Loss:[total:{:.4f}, c:{:.4f}, o:{:.4f}, co:{:.4f}] | Train:[c:{:.4f}, o:{:.4f}, co:{:.4f}]'
                    .format(epoch, 
                            args.epochs, 
                            iter, 
                            len(data_loader), 
                            epoch_loss / (iter + 1), 
                            epoch_loss_c / (iter + 1), 
                            epoch_loss_o / (iter + 1), 
                            epoch_loss_co / (iter + 1), 
                            epoch_train_acc_c / nb_data * 100,
                            epoch_train_acc_o / nb_data * 100,
                            epoch_train_acc_co / nb_data * 100))
    
    epoch_loss /= (iter + 1)
    epoch_loss_c /= (iter + 1)
    epoch_loss_o /= (iter + 1)
    epoch_loss_co /= (iter + 1)
    epoch_train_acc_o /= nb_data
    loss_dict["total"] = epoch_loss
    loss_dict["c"] = epoch_loss_c
    loss_dict["o"] = epoch_loss_o
    loss_dict["co"] = epoch_loss_co

    return loss_dict, epoch_train_acc_o, optimizer



def evaluate_network(model, device, data_loader, epoch, args):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc_co = 0
    epoch_test_acc_c = 0
    epoch_test_acc_o = 0
    nb_data = 0
    acc_dict = {}
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            
            xc, xo, xco = model.forward(batch_graphs, batch_x, batch_e, args.eval_random)
            loss = model.loss(xo, batch_labels) 
            epoch_test_loss += loss.detach().item()

            epoch_test_acc_co += accuracy(xco, batch_labels)
            epoch_test_acc_c += accuracy(xc, batch_labels)
            epoch_test_acc_o += accuracy(xo, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc_co /= nb_data
        epoch_test_acc_c /= nb_data
        epoch_test_acc_o /= nb_data
    
    acc_dict['co'] = epoch_test_acc_co
    acc_dict['c'] = epoch_test_acc_c
    acc_dict['o'] = epoch_test_acc_o
    return epoch_test_loss, acc_dict

