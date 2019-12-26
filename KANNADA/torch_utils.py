import os, sys, random, math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import itertools as it
import scipy
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer

from tqdm import tqdm_notebook as tqdm

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage

from collections import OrderedDict

from sklearn import preprocessing
import gc


def is_interactive():
    ''' Return True if inside a notebook/kernel in Edit mode, or False if committed '''
    return 'runtime' in get_ipython().config.IPKernelApp.connection_file

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def train_classifier(model, optimizer, criterion, current_epoch, train_loader, device="cpu", print_interval=10, custom_txf=None, one_hot=False):
    model.train()
    model.to(device)
    
    train_correct = 0
    train_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        len_data = len(data)
        len_dataset = len(train_loader.dataset)
        len_loader = len(train_loader)
        
        
        data, target = data.to(device), target.to(device)
        
        if custom_txf is not None:
            data, target = custom_txf(data, target)
        
        output = model(data)
        
        if not one_hot:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            pred = output.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()
        
        loss = criterion(output, target)
        train_loss+= (loss.item() * len_data)
        
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % print_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.8f}%)]\tLoss: {:.8f}'.format(
                    current_epoch, batch_idx * len_data, len_dataset,100. * batch_idx / len_loader, loss.item()
                    )
                )
    ## This is training, so reduction = mean, i.e. loss.item() already gives the mean of the batch
    train_loss/=len(train_loader.dataset)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    print('Train Set: Average loss: {:.8f}, Accuracy: {}/{} ({:.8f}%)'.format(
        train_loss, train_correct, len(train_loader.dataset), train_accuracy
        )
    )
    
    return train_loss, train_accuracy
    
    
    
def test_classifier(model, criterion, device, test_loader, one_hot=False, tta=False):
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if tta:
                batch_size, n_crops, c, h, w = data.size()
                data = data.view(-1, c, h, w)
                output = model(data)
                output = output.view(batch_size, n_crops, -1).mean(1)
            else:
                output = model(data)
            
            test_loss += criterion(output, target).item()  # sum up batch loss
            if not one_hot:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                test_correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                pred = output.data.max(1, keepdim=True)[1]
                test_correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()


    ## validation/test, so reduction = "sum"
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.8f}%)\n'.format(
        test_loss, test_correct, len(test_loader.dataset),test_accuracy
        )
    )
    return test_loss, test_accuracy





class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """Coding Credit: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, track="min", patience=7, verbose=False, delta=0):
        """
        Args:
            track (str): What to track, possible value: min, max (e.g. min validation loss, max validation accuracy (%))
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.track = track
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = -np.Inf
        if self.track=="min":
            self.val_best = np.Inf
        self.delta = delta

    def __call__(self, current_score, model):

        score = -current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif ((score < self.best_score + self.delta) and self.track=="min") or ((score>self.best_score+self.delta) and self.track=="max"):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, new_best, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Found better solution ({self.val_best:.6f} --> {new_best:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_best = new_best


class LabelSmoothingLoss(nn.Module):
    '''Always returns mean of the loss'''
    """https://github.com/pytorch/pytorch/issues/7455"""
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
    


def one_hot_cross_entropy(y_logit, y_target, reduction="mean"):
    """Calculates CE loss for one hot target. 1e-10 for numerical stability"""
    """
        Args:
            y_logit: logit output of the network.
            y_target: one hot encoded target
            reduction: "mean" or "sum"        
    """
    assert (reduction=="mean") or (reduction=="sum")
    
    loss = -(y_target*torch.log_softmax(y_logit, dim=1) + 1e-10).sum(dim=1)
    
    if reduction=="mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class CNNLayer(nn.Module):
    def __init__(self, input_depth, output_depth, kernel_size=3):
        super(CNNLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(input_depth, output_depth, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(output_depth),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layer(x)

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True)
        )
        
        self.residual = (input_size==output_size)
    
    def forward(self, x):
        out = self.layer(x)
        if self.residual:
            return (out+x)/np.sqrt(2)
        return out
    
    
def manifold_mixup(x, shuffle, lmbda, i, j):
    if shuffle is not None and lmbda is not None and i==j:
        x = lmbda*x+(1-lmbda)*x[shuffle]
    return x


def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    

class Sq_Ex_Block(nn.Module):
    def __init__(self, in_ch, r):
        super(Sq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        return x.view(*(x.shape[:-2]),-1).mean(-1)
    
    
def cos_annealing_lr(init_lr, cur_epoch, epoch_per_cycle):
    return init_lr*(np.cos(np.pi*cur_epoch/epoch_per_cycle)+1)/2

def xavier_init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)