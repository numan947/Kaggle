import os, sys, random, math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
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

import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
                    
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)


                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

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
            
            

def resumetable(df):
    """https://www.kaggle.com/kabure/eda-feat-engineering-encode-conquer"""
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(scipy.stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    """https://www.kaggle.com/kabure/eda-feat-engineering-encode-conquer"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model', random_state=947):
    """https://www.kaggle.com/peterhurford/why-not-logistic-regression"""
    """"
        Example:
        from sklearn.linear_model import LogisticRegression
        def runLR(train_X, train_y, test_X, test_y, test_X2, params):
            print('Train LR')
            model = LogisticRegression(**params)
            model.fit(train_X, train_y)
            print('Predict 1/2')
            pred_test_y = model.predict_proba(test_X)[:, 1]
            print('Predict 2/2')
            pred_test_y2 = model.predict_proba(test_X2)[:, 1]
            return pred_test_y, pred_test_y2


        lr_params = {'solver': 'lbfgs', 'C': 0.1}
        results = run_cv_model(ohTrainX, ohTestX, trainY, runLR, lr_params, metrics.roc_auc_score, 'lr')
    """
    kf = ms.KFold(n_splits=5, random_state=random_state)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
        dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores}

    return results



from sklearn.base import TransformerMixin
from itertools import repeat
import scipy


class ThermometerEncoder(TransformerMixin):
    """
    kaggle.com/superant/oh-my-cat
    Assuming all values are known at fit.
    Example:
    thermos = []
    for col in  ["ord_0","ord_1","ord_2","ord_3","ord_4","ord_5a","ord_5b"]:
        if col=="ord_0":
            sort_key = int
        elif col=="ord_1":
            sort_key = ["Novice", "Contributor", "Expert", "Master", "Grandmaster"].index
        elif col=="ord_2":
            sort_key = ["Freezing","Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"].index
        elif col in ["ord_3", "ord_4", "ord_5a", "ord_5b"]:
            sort_key = str
        elif col in ["day", "month"]:
            sort_key = int
        else:
            raise ValueError(col)
        print(col)
        enc = ThermometerEncoder(sort_key)
        thermos.append(enc.fit_transform(catted[col]))        
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val:i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        possible_values = sorted(self.value_map_.values())
        idx1,idx2 = [],[]
        
        all_indices = np.arange(len(X))
        for idx, val  in enumerate(possible_values[:-1]):
            new_idx = all_indices[values>val]
            idx1.extend(new_idx)
            idx2.extend(repeat(idx, len(new_idx)))
        
        result = scipy.sparse.coo_matrix(([1]*len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
        return result




#########################################################################################
#https://www.kaggle.com/adaubas/2nd-place-solution-categorical-fe-callenge
def color_and_top(nb_mod, feature, typ, top_n=None):
    
    if top_n is None:
        resu = ["g", nb_mod]
    elif nb_mod > 2*top_n:
        resu = ["r", top_n]
    elif nb_mod > top_n:
        resu =["orange", top_n]
    else: 
        resu = ["g", nb_mod]
    
    title = feature[:20]+" ("+typ[:3]+"-{})".format(nb_mod)
    resu.append(title)
    
    return resu


def plot_multiple_categorical(df, features, col_target=None, top_n=None
                              , nb_subplots_per_row = 4, hspace = 1.3, wspace = 0.5
                              , figheight=15, m_figwidth=4.2, landmark = .01):
    
    sns.set_style('whitegrid')
    
    if not (col_target is None):
        ref = df[col_target].mean() # Reference
    
    plt.figure()
    if len(features) % nb_subplots_per_row >0:
        nb_rows = int(np.floor(len(features) / nb_subplots_per_row)+1)
    else:
        nb_rows = int(np.floor(len(features) / nb_subplots_per_row))
    fig, ax = plt.subplots(nb_rows, nb_subplots_per_row, figsize=(figheight, m_figwidth * nb_rows))
    plt.subplots_adjust(hspace = hspace, wspace = wspace)

    i = 0; n_row=0; n_col=0
    for feature in features:
        
        i += 1
        plt.subplot(nb_rows, nb_subplots_per_row, i)

        dff = df[[feature, col_target]].copy() # I don't want transform data, only study them
        
        # Missing values
        if dff[feature].dtype.name in ["float16", "float32", "float64"]:
            dff[feature].fillna(-997, inplace=True)
            
        if dff[feature].dtype.name in ["object"]:
            dff[feature].fillna("_NaN", inplace=True)
            
        if dff[feature].dtype.name == "category" and dff[feature].isnull().sum() > 0:
            dff[feature] = dff[feature].astype(str).replace('', '_NaN', regex=False).astype("category")
            
        # Colors, title
        bar_colr, top_nf, title = color_and_top(dff[feature].nunique(), feature, str(dff[feature].dtype), top_n)
        
        # stats
        tdf = dff.groupby([feature]).agg({col_target: ['count', 'mean']})
        tdf = tdf.sort_values((col_target, 'count'), ascending=False).head(top_nf).sort_index()
        
        tdf.index = tdf.index.map(str)
        tdf = tdf.rename(index={'-997.0':'NaN'}) # Missing values
        if not (top_n is None):
            tdf.index = tdf.index.map(lambda x: x[:top_n]) # tronque les libellés des modalités en abcisse
        
        tdf["ref"] = ref
        tdf["ref-"] = ref-landmark
        tdf["ref+"] = ref+landmark
        
        # First Y axis, on the left
        plt.bar(tdf.index, tdf[col_target]['count'].values, color=bar_colr) # Count of each category
        
        plt.title(title, fontsize=11)
        plt.xticks(rotation=90)
        
        # Second Y axis, on the right
        xx = plt.xlim()
        if nb_subplots_per_row == 1:
            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex = ax[n_row], frameon = False)
        else:
            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex = ax[n_row, n_col], frameon = False)
        if not (col_target is None):
            ax2.plot(tdf[col_target]['mean'].values, marker = 'x', color = 'b', linestyle = "solid") # Mean of each Category
            ax2.plot(tdf["ref"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=4.0) # Reference
            ax2.plot(tdf["ref-"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=1.0) # Reference
            ax2.plot(tdf["ref+"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=1.0) # Reference
        ax2.yaxis.tick_right()
        ax2.axes.get_xaxis().set_visible(False)
        plt.xlim(xx)

        n_col += 1
        if n_col == nb_subplots_per_row:
            n_col = 0
            n_row += 1
            
    plt.show();
#########################################################################################

## Count trainable parameters for a PyTorch model
def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





## ALL IMPORTS FOR A NEW NOTEBOOK

import os, sys, random, math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import itertools as it
import scipy
import glob
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Optimizer
import torchvision.transforms.transforms as txf
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict

from sklearn import metrics
from sklearn import preprocessing as pp
from sklearn import model_selection as ms

import torch_utils
from tqdm.notebook import tqdm_notebook as tqdm
import time

font = {'size'   : 20}

matplotlib.rc('font', **font)


###########################################################################
# MAP CATEGORICAL NOMINAL VALUES WHICH ARE NOT IN BOTH TRAIN AND TEST TO A SINGLE UNIQUE VALUE
# for col in range(5,10):
#     col = "nom_"+str(col)
#     train_uniq = set(train_df[col].unique())
#     test_uniq = set(test_df[col].unique())
    
#     xor_cat = train_uniq ^ test_uniq
    
#     if len(xor_cat)>0:
#         catted.loc[catted[col].isin(xor_cat), col] = "xor"

###########################################################################