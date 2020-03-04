import os, sys, random, math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools as it
import scipy
import glob
import datetime
import pickle

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



def load_experiments():
    RESULT_FILE_NAME = "./EXPERIMENTS.csv"
    if os.path.exists(RESULT_FILE_NAME):
        return pd.read_csv(RESULT_FILE_NAME, index_col="name")
    else:
        f = open(RESULT_FILE_NAME, "w")
        f.write("name,result1,result2,result3")
        f.close()
        return load_experiments()
def save_experiments(df):
    RESULT_FILE_NAME = "./EXPERIMENTS.csv"
    df.to_csv(RESULT_FILE_NAME, index=False)

def load_pickled_object(path):
    with open(path, 'rb') as config_dictionary_file:
        return pickle.load(config_dictionary_file)
        
def pickle_save_object(dictionary, path):
    with open('config.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(dictionary, config_dictionary_file)    


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
    def __init__(self, track="min", patience=7, verbose=False, delta=0, save_model_name="checkpoint.pt"):
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
        self.save_model_name = save_model_name

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
        torch.save(model.state_dict(), self.save_model_name)
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

def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*scale
    return optimizer
    
    
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
def get_color_top_n_title(feature_unique_count, feature_name, feature_type, top_n=None):
    if top_n is None:
        result = ["g", feature_unique_count]
    elif feature_unique_count > 2*top_n:
        result = ["r", top_n]
    elif feature_unique_count > top_n:
        result =["y", top_n]
    else: 
        result = ["g", top_n]
    
    title = feature_name[:20]+" ("+feature_type[:3]+"-{})".format(feature_unique_count)
    result.append(title)
    return result

def plot_multiple_categorical(df, features, col_target=None, top_n=None, 
                              n_subplots_per_row=4, hspace=1.3, wspace=0.5,
                              fig_h=15, m_fig_w=4.2, landmark=0.01, save=False):
    
    if col_target is not None:
        ref = df[col_target].mean()
        tgtFeat = df[col_target].copy()
    
    
    plt.figure()
    
    total_rows = int(np.ceil(1.0*len(features)/n_subplots_per_row))
    
    fig, ax = plt.subplots(total_rows, n_subplots_per_row, figsize=(fig_h, m_fig_w*total_rows))
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    
    for i, f in enumerate(features):
        curFeat = df[f].copy()
        curFeatDtype = curFeat.dtype.name
        
        ## Take care of the missing values
        if curFeatDtype in ["float16","float32","float64"]:
            curFeat.fillna(-1000.0, inplace=True)
        if curFeatDtype in ["object"]:
            curFeat.fillna("_NaN", inplace=True)
        if curFeatDtype == "category" and curFeat.isnull().sum() > 0:
            curFeat = curFeat.astype(str).replace('', '_NaN', regex=False).astype("category")
        
        
        ## create bar color
        bar_color, top_ns, title = get_color_top_n_title(curFeat.nunique(), f, str(curFeat.dtype.name), top_n)
        
        ## create statistics
        
        if col_target is not None:
            tmpdf = pd.DataFrame()
            tmpdf[col_target] = tgtFeat
            tmpdf[f]=curFeat
            
            stats = tmpdf.groupby([f]).agg({col_target: ['count', 'mean']})
            stats = stats.sort_values((col_target, 'count'), ascending=False).head(top_ns).sort_index()
        
            stats.index = stats.index.map(str)
            stats = stats.rename(index={'-1000.0':'NaN'})
            if top_n is not None:
                stats.index = stats.index.map(lambda x: x[:top_n])
        
            stats["ref"] = ref
            stats["ref-"] = ref-landmark
            stats["ref+"] = ref+landmark
            ax[i//n_subplots_per_row][i%n_subplots_per_row].bar(stats.index, stats[col_target]['count'], color=bar_color)
            
            xx = ax[i//n_subplots_per_row][i%n_subplots_per_row].get_xlim()
            
            ax2 = fig.add_subplot(total_rows,n_subplots_per_row, i+1, sharex=ax[i//n_subplots_per_row,i%n_subplots_per_row], frameon=False)
            ax2.plot(stats[col_target]['mean'].values, marker="x", color="b", linestyle="dashed")
            
            ax2.plot(stats["ref"].values, marker="_", color="black", linestyle="solid", linewidth=2.5)
            ax2.plot(stats["ref-"].values, marker="_", color="black",linewidth=1)
            ax2.plot(stats["ref+"].values, marker="_", color="black",linewidth=1)
            
            ax2.yaxis.tick_right()
            ax2.axes.get_xaxis().set_visible(False)
            
            ax[i//n_subplots_per_row][i%n_subplots_per_row].set_xlim(xx)
            
            
        else:
            vc = curFeat.value_counts()
            vc = vc.head(top_ns).sort_index()
            vc.index = vc.index.map(str)
            vc = vc.rename(index={"-1000.0":"NaN"})
            
            if top_n is not None:
                vc.index = vc.index.map(lambda x: x[:top_n])
            ax[i//n_subplots_per_row][i%n_subplots_per_row].bar(vc.index, vc.values, color=bar_color)
        
        ax[i//n_subplots_per_row][i%n_subplots_per_row].set_title(title, fontsize=12)
        ax[i//n_subplots_per_row][i%n_subplots_per_row].tick_params(axis="x",rotation=90)
    
    
    if(n_subplots_per_row*total_rows!=len(features)):
        for i in range(1,1+(n_subplots_per_row*total_rows)-len(features)):
            ax[-1][-i].axis("off")
    
    fig.tight_layout()
    if save:
        plt.savefig("CatPlot showing relation with target.png", bbox_inches="tight")
    else:
        plt.show()
#########################################################################################
## Count trainable parameters for a PyTorch model
def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_epoch_stat(epoch_idx, time_elapsed_in_seconds, history=None, train_loss=None, train_accuracy=None, valid_loss=None, valid_accuracy=None):
    print("\n\nEPOCH {} Completed, Time Taken: {}".format(epoch_idx+1, datetime.timedelta(seconds=time_elapsed_in_seconds)))
    if train_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "train_loss"] = train_loss
        print("\tTrain Loss \t{:0.9}".format(train_loss))
    if train_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "train_accuracy"] = 100.0*train_accuracy
        print("\tTrain Accuracy \t{:0.9}%".format(100.0*train_accuracy))
    if valid_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_loss"] = valid_loss
        print("\tValid Loss \t{:0.9}".format(valid_loss))
    if valid_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_accuracy"] = 100.0*valid_accuracy
        print("\tValid Accuracy \t{:0.9}%".format(100.0*valid_accuracy))
    
    return history


"""
CODE TAKEN FROM: https://github.com/pytorch/text/blob/master/torchtext/data/metrics.py
"""

import math
import collections
import torch
from torchtext.data.utils import ngrams_iterator

def _compute_ngram_counter(tokens, max_n):
    """ Create a Counter with a count of unique n-grams in the tokens list
    Arguments:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted
    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count
    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1,
             ('me', 'me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(' '))
                                         for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf
    Arguments:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)
    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['I', 'ate', 'the', 'apple'], ['I', 'did']]
        >>> references_corpus = [[['I', 'ate', 'it'], ['I', 'ate', 'apples']],
                [['I', 'did']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.7598356856515925
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(references_corpus),\
        'The length of candidate and reference corpus should be the same'

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(len(candidate) - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for ngram in candidate_counter:  # TODO: no need to loop through the whole counter
            total_counts[len(ngram) - 1] += candidate_counter[ngram]

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()

class PositionalEncoding(nn.Module):
    """Implement the PE function of the iconic paper: "Attention is all you need". 
    This is basically a embedding layer for the positions. It supports upto 5000 positions.
    Input the word embeddings, and it will return the added final embeddings.    
    """
    def __init__(self, d_model, dropout=0.4, max_len=5000):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)



def summarize_all(df):
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




# contingency table should be a dataframe
# cramerV is kind of correlation between two categorical variables
def calculate_cramerV(contingency_table):
    # get the chi2 value
    chi2 = scipy.stats.chi2_contingency(contingency_table)[0]
    # count total values in the table
    n = contingency_table.sum().sum()
    # get the number of rows and columns in the table
    r,c = contingency_table.shape
    # calculate cramerV
    cramerV = np.sqrt(chi2/(n*min(r-1, c-1)))
    
    return cramerV

def summarize_categorical(train, test, use_feat=[], exclude_feat=[], target_for_cramerV=None):
    """
    train: train dataframe
    test: test dataframe
    
    use_feat: list of features for which statistics will be calculated, if empty, uses all columns except the ones in exclue_feat
    exclue_feat: features to be exclueded from the statistics calculation
    
    target_for_cramerV: target column label for cramerV calculation

    """
    
    
    all_stats = []
    
    if len(use_feat)==0:
        use_feat = [c for c in list(train.columns) if c not in exclude_feat]
    
    for col in tqdm(use_feat):
        vcTrain = dict(train[col].value_counts())
        vcTest = dict(test[col].value_counts())
        
        set_train_only_vals = set(vcTrain.keys())-set(vcTest.keys())
        set_test_only_vals = set(vcTest.keys())-set(vcTrain.keys())
        
        dict_train_only_vals = {k:v for k,v in vcTrain.items() if k in set_train_only_vals}
        dict_test_only_vals = {k:v for k,v in vcTest.items() if k in set_test_only_vals}
        
        trainUniqueVals, trainTotalVals = len(vcTrain), pd.Series(vcTrain).sum()
        trainOnlyUniqueVals, trainOnlyTotalVals = len(dict_train_only_vals), pd.Series(dict_train_only_vals).sum()
        
        testUniqueVals, testTotalVals = len(vcTest), pd.Series(vcTest).sum()
        testOnlyUniqueVals, testOnlyTotalVals = len(dict_test_only_vals), pd.Series(dict_test_only_vals).sum()
        
        
        if target_for_cramerV is not None:
            contingency_table = pd.crosstab(train[col], train[target_for_cramerV].fillna(-1))
            vc = calculate_cramerV(contingency_table)
        else:
            vc = -1
            
        all_stats.append((
            col, round(vc, 3), train[col].nunique(),
            test[col].nunique(),
            str(trainOnlyUniqueVals)+"("+str(round(100.0*(trainOnlyUniqueVals/trainUniqueVals),3))+")",
            str(testOnlyUniqueVals)+"("+str(round(100.0*(testOnlyUniqueVals/testUniqueVals),3))+")",
            str(train[col].isnull().sum())+"("+str(round(100.0*(train[col].isnull().sum()/train.shape[0]),3))+")",
            str(train[col].isnull().sum())+"("+str(round(100.0*(train[col].isnull().sum()/train.shape[0]),3))+")",
            str(train[col].value_counts().index[0])+"("+str(round(100.0 * train[col].value_counts(normalize = True, dropna = False).values[0], 3))+")",
            train[col].dtype
        ))
    
    
    df_stats = pd.DataFrame(all_stats,columns=[
        "Feature",
        "Target Cramer's V",
        "Unique values (Train)",
        "Unique values (Test)",
        "Train only value counts",
        "Test only value counts",
        "Missing (Train)",
        "Missing (Test)",
        "Value with the highest counts (Train)",
        "DataType"
    ])
    
    if target_for_cramerV is None:
        df_stats.drop(["Target Cramer's V"], axis=1, inplace=True)
        
    return df_stats, dict_train_only_vals, dict_test_only_vals

########################################################## OBJECT DETECTION UTILS ################################################################################################################

def xy_to_cxcy(xy):
    """
    Convert BBoxes from (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height)
    """
    return torch.cat([ (xy[:, 2:] + xy[:, :2])/2.0, (xy[:, 2:] - xy[:,:2])], dim=1)

def cxcy_to_xy(cxcy):
    """
    Convert BBoxes from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
    """
    return torch.cat([(cxcy[:, :2]-(cxcy[:,2:]/2.0)), (cxcy[:,:2]+(cxcy[:,2:]/2.0))], dim=1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode (center_x, center_y, width, height) to difference wrt a prior
    """
    return torch.cat([(cxcy[:,:2] - priors_cxcy[:,:2])/(priors_cxcy[:,2:]/10), torch.log(cxcy[:,2:]/priors_cxcy[:,2:]) * 5], dim=1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode encoded coordinates to (center_x, center_y, width, height) using a prior
    """
    return torch.cat([ (gcxgcy[:,:2]*(priors_cxcy[:,2:]/10))+priors_cxcy[:,:2], (torch.exp(gcxgcy[:,2:]/5)*priors_cxcy[:,2:]) ], dim=1)



import torchvision.transforms.functional as FT

def box_resize(image, boxes, dims=(300,300), return_percent_coords=True):
    """
    Resizing Image in Object Detection context.
    Given, image, object bboxes and dims to resize, this function resizes the image to the specified
    dimensions and adjusts the positions of the boxes according to the resized image
    
    """
    new_image = FT.resize(image, dims)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes/old_dims
    
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes*new_dims
        
    
    return new_image, new_boxes

def box_flip(image, boxes):
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:,0]
    new_boxes[:, 2] = image.width - boxes[:,2]
    new_boxes = new_boxes[:, [2,1,0,3]]
    return new_image, new_boxes

def box_expand(image, boxes, filler):
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(original_h*scale)
    new_w = int(original_w*scale)
    
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)
    
    left = random.randint(0, new_w-original_w)
    right = left+original_w
    
    top = random.randint(0, new_h-original_h)
    bottom = top+original_h
    
    new_image[:,top:bottom, left:right] = image
    
    new_boxes = boxes+torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    
    return new_image, new_boxes

def box_random_crop(image, boxes, labels, difficulties):
    original_h = image.size(1)
    original_w = image.size(2)
    
    while True:
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])
        
        if min_overlap is None:
            return image, boxes, labels, difficulties
        
        max_trials = 50
        
        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1.0)
            scale_w = random.uniform(min_scale, 1.0)
            
            new_h = int(scale_h*original_h)
            new_w = int(scale_w*original_w)
            
            aspect_ratio = new_h/new_w
            
            if not(0.5<aspect_ratio<2.0):
                # end this trial
                continue
            
            left = random.randint(0, original_w-new_w)
            right = left+new_w
            top = random.randint(0, original_h-new_h)
            bottom = top+new_h
            
            crop = torch.FloatTensor([left, top, right, bottom])
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)
            
            if overlap.max().item()< min_overlap:
                # end this trial
                continue
            
            new_image = image[:, top:bottom, left:right]
            
            bb_centers = (boxes[:,:2]+boxes[:,2:])/2.0
            
            centers_in_crop = (bb_centers[:, 0]>left)*(bb_centers[:,0]<right)*(bb_centers[:,1]>top)*(bb_centers[:,1]<bottom)
            
            if not centers_in_crop.any():
                # we have some how produced a crop not containing any objects!!
                continue
            
            new_boxes = boxes[centers_in_crop,:]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]
            
            
            new_boxes[:,:2] = torch.max(new_boxes[:,:2], crop[:2])
            new_boxes[:,:2] -= crop[:2]
            new_boxes[:,2:] = torch.min(new_boxes[:,2:], crop[2:])
            new_boxes[:,2:] -= crop[:2]
            
            return new_image, new_boxes, new_labels, new_difficulties            
def photometric_distort(image):
    """
    Distorts brightness, contrast, saturation and hue, each with 50% chance in random order
    :param image: PIL image
    """
    
    new_image = image
    
    distortions = [FT.adjust_brightness, FT.adjust_contrast, FT.adjust_saturation, FT.adjust_hue]
    
    random.shuffle(distortions)
    
    for d in distortions:
        if random.random()<0.5:
            if d.__name__ is "adjust_hue":
                adjust_factor = random.uniform(-18/255.0, 18/255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = d(new_image, adjust_factor)
    
    return new_image


def find_intersection(set_1, set_2):
    """
    Finds intersection between two sets of boxes
    (n1,4) and (n2, 4)
    returns (n1,n2)
    """
    
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(dim=1), set_2[:,:2].unsqueeze(dim=0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(dim=1), set_2[:,2:].unsqueeze(dim=0))
    
    intersection = torch.clamp(upper_bounds-lower_bounds, min=0)
    
    return intersection[:,:,0]*intersection[:,:,1]


def find_jaccard_overlap(set_1, set_2):
    """
    Jaccard overlap between two sets of boxes.
    :param set_1: a tensor of dimension (n1, 4), where n1 is number of boxes each having 4 coordinates
    :param set_2: a tensor of dimension (n2, 4), where n2 is number of boxes each having 4 coordinates
    Coordinates are in (x1, y1, x2, y2) form, representing (top, left) and (bottom, right) corners
    :return: a tensor of dimension (n1, n2) where each value represent the jaccard overlap between two corresponding boxes
    """
    
    intersection = find_intersection(set_1, set_2)
    areas_set_1 = (set_1[:, 2] - set_1[:,0])*(set_1[:,3]-set_1[:,1])
    areas_set_2 = (set_2[:, 2] - set_2[:,0])*(set_2[:,3]-set_2[:,1])
    
    union = areas_set_1.unsqueeze(1)+areas_set_2.unsqueeze(0)-intersection
    
    return intersection/union


def decimate(tensor, m):
    """
    Decimate a tensor by a factor of 'm', i.e. downsample by keeping every m'th value.
    This is used to convert FC layers to equivalent Conv layers which are smaller in size.
    
    :param tensor: tensor to be decimated
    :m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    
    assert(tensor.dim() == len(m))
    
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(0, tensor.size(d), m[d]).long())
        
    
    return tensor


def clip_gradients(model, clip_val):
    nn.utils.clip_grad_norm_(model.parameters(), clip_val)


def save_checkpoint(epoch, model, optimizer):
    state = {
        "epoch":epoch,
        "state_dict":model.state_dict(),
        "optimizer":optimizer
    }
    filename = "Checkpoint.pth.tar"
    torch.save(state, filename)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


##################################################################################################################################################################
################################################## RIGOROUS CROSS VALIDATION #####################################################################################
# fold1 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED)
# fold2 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED+3)
# fold3 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED+5)

# def cross_val_print(pipe, X, y, cv, scoring="roc_auc", best_score=0.):
#     scores = ms.cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=True)
#     cv_score = scores["test_score"].mean()
#     train_score = scores["train_score"].mean()
    
#     if cv == fold1:
#         precision=1
#     elif cv == fold2:
#         precision=2
#     elif cv == fold3:
#         precision=3
        
    
#     print("CV{} score on valid: {:.7f} - Previous best valid score: {:.7f} - Train mean score: {:6f}".format(precision, cv_score, best_score, train_score))
    
#     if cv_score>best_score:
#         best_score = cv_score
    
#     return cv_score, best_score
##################################################################################################################################################################
################################################## TRANSFORMING CATEGORICAL TO ORDINAL ###########################################################################

# def transform_feat(srs):
#     transform_dict = {
#         "ord_1":{"Novice":0,"Contributor":1,"Expert":2, "Master":3, "Grandmaster":4},
#         "ord_2":{"Freezing":0,"Cold":1,"Warm":2, "Hot":3, "Boiling Hot":4,"Lava Hot":5},
#         "nom_0":{"Blue":1,"Green":2, "Red":3},
#         "nom_1":{"Circle":1,"Trapezoid":2, "Star":3,"Polygon":4,"Square":5,"Triangle":6},
#         "nom_2":{"Dog":1,"Lion":2, "Snake":3,"Axolotl":4,"Cat":5,"Hamster":6},
#         "nom_3":{"Finland":1,"Russia":2, "China":3,"Costa Rica":4,"Canada":5,"India":6},
#         "nom_4":{"Bassoon":1,"Piano":2, "Oboe":3,"Theremin":4}   
#     }
    
#     if srs.name == "ord_0":
#         return srs-1
#     elif srs.name == "ord_5":
#         vals = list(np.sort(srs.unique()))
#         return srs.map({l:i for i, l in enumerate(vals)})
#     elif srs.name in ["ord_3", "ord_4"]:
#         return srs.str.lower().map({l:i for i, l in enumerate(list(ascii_lowercase))})
#     elif srs.name in transform_dict.keys():
#         return srs.map(transform_dict[srs.name])
#     else:
#         return srs
#
##################################################################################################################################################################
######################################################## BINNING HIGH CARDINALITY FEATURE #######################################################################
# class BinsEncoder(BaseEstimator, TransformerMixin):
#     """
#         Binning high cardinality categorical values.
        
#     """
#     def __init__(self, nbins=200, nmin=20):
#         self.nbins = nbins
#         self.nmin = nmin
    
#     def fit(self, X, y=None):
#         tmp = pd.concat([X,y], axis=1)
        
#         averages = tmp.groupby(by=X.name)[y.name].mean()
#         means_for_each_vals = dict(zip(averages.index.values, averages.values))
#         bins = np.linspace(averages.min(), averages.max(), self.nbins)
#         self.map_ = dict(zip(averages.index.values, np.digitize(averages.values, bins=bins)))
        
#         ## If some key has more than nmin observations, keep the original key, otherwise bin the key
#         count = tmp.groupby(by=X.name)[y.name].count()
#         count_for_each_vals = dict(zip(count.index.values, count.values))
        
#         for key, val in count_for_each_vals.items():
#             if val>self.nmin:
#                 self.map_[key]=key
            
#         return self
    
#     def transform(self, X, y=None):
#         tmp = X.map(self.map_)
#         tmp.fillna(random.choice(list(self.map_.values())), inplace=True)
#         tmp = tmp.astype(str)
#         return tmp
##################################################################################################################################################################
################################################## FEATURE ENGINEERING WITH BINNING AND ORDINAL TRANSFORM ########################################################
# class FEUpgraded(BaseEstimator, TransformerMixin):
#     def __init__(self, list_ordinal_features=[], feat_to_bins_encode={}):
#         self.list_ordinal_features = list_ordinal_features
#         self.feat_to_bins_encode = feat_to_bins_encode
#         self.BinsEncoder = {}
    
#     def fit(self, X, y=None):
#         for feat, val in self.feat_to_bins_encode.items():
#             self.BinsEncoder[feat] = BinsEncoder(nbins=val[0], nmin=val[1])
#             self.BinsEncoder[feat].fit(X[feat], y)
#         return self
#     def transform(self, X, y=None):
#         df = X.copy()
#         for v in self.feat_to_bins_encode.keys():
#             df[v] = self.BinsEncoder[v].transform(df[v])
            
#         for v in self.list_ordinal_features:
#             df[v] = transform_feat(df[v])
        
#         return df



# class CrossValTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, transformer, cv):
#         self.transformer = transformer
#         self.cv = cv
#     def fit(self, X, y=None):
#         self.mode = X[self.columns].mode().values[0]
#         return self
#     def transform(self, X, y=None):
#         df = X.copy()
#         for i, col in enumerate(self.columns):
#             df[col].fillna(self.mode[i],inplace=True)
#         return df
#     def get_oof_transform(X, y):
#         oof = pd.DataFrame(index=X.index, columns=X.columns)
#         for train_idx, valid_idx in self.cv.split(X,y):
#             train_X = X.loc[train_idx]
#             train_y = y.loc[train_idx]
#             valid_X = X.loc[valid_idx]
#             valid_y = y.loc[valid_idx]
            
#             self.transformer.fit(train_X, train_y)
#             oof_txf = self.transformer(valid_X)
            
#             oof.loc[valid_idx] = oof_txf
#         return oof



# def preprocessor(X_train, y_train, feat1, txf1, feat2, txf2, feat3, txf3, feat4, txf4, feat5, txf5, verbose=False):
#     if(verbose):
#         print("Preprocessing....")
#     st = time.time()
    
#     preprocessed_x_1 = txf1.fit_transform(X_train[feat1], y_train).astype('float') # ordinal
#     preprocessed_x_2 = txf2.fit_transform(X_train[feat2], y_train) # one hot
#     preprocessed_x_3 = txf3.get_oof_transform(X_train[feat3], y_train).astype("float") # trg
#     preprocessed_x_4 = txf4.get_oof_transform(X_train[feat4], y_train).astype("float") # cat
#     preprocessed_x_5 = txf5.get_oof_transform(X_train[feat5], y_train).astype("float") # woe
    
#     # merge preprocessed
#     PPX_train = scipy.sparse.hstack([
#         preprocessed_x_1, preprocessed_x_2,
#         preprocessed_x_3, preprocessed_x_4,
#         preprocessed_x_5]).tocsr()
    
#     if verbose:
#         print("Preprocessing....Done -- Time Taken: {}".format(datetime.timedelta(seconds=time.time()-st)))
#     return PPX_train, y_train

# def cross_val_train_f(model, X_train, y_train, cv, score_function, best_score, verbose=False):
#     start_time = time.time()
    
#     valid_scores = []
#     train_scores = []
#     for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
#         if verbose:
#             print("Training....Fold {}".format(i+1))
#         train_X = X_train[train_idx]
#         train_y = y_train.loc[train_idx]
        
#         valid_X = X_train[valid_idx]
#         valid_y = y_train.loc[valid_idx]
        
#         # fit and score
#         model.fit(train_X, train_y)
#         train_score = score_function(train_y, model.predict_proba(train_X)[:,1])
#         valid_score = score_function(valid_y, model.predict_proba(valid_X)[:,1])
        
#         if verbose:
#             print("Fold {} train score: {:0.5f}".format(i+1, train_score))
#             print("Fold {} valid score: {:0.5f}".format(i+1, valid_score))
        
        
#         train_scores.append(train_score)
#         valid_scores.append(valid_score)
    
#     cv_score = np.array(valid_scores).mean()
#     train_score = np.array(train_scores).mean()
    
#     down = '\u2193'
#     up = "\u2191"
#     curarr = ""
#     if (cv_score-best_score)>0:
#         curarr = up
#     elif (cv_score-best_score)<0:
#         curarr = down

#     print("{}CV valid score: {:.7f} - Previous best score: {:.7f} - Train score: {:6f} - Time {}".format(curarr, cv_score, best_score, train_score, str(datetime.timedelta(seconds=time.time()-start_time))))
    
#     return cv_score
##################################################################################################################################################################



## ALL IMPORTS FOR A NEW NOTEBOOK
__SEED = 0
__N_FOLDS = 5
__NROWS = None

import os, sys, random, math
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
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

import ml_utils as mu
import time
import time, datetime, pickle


# fold1 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED)
# fold2 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED+3)
# fold3 = ms.StratifiedKFold(n_splits=__N_FOLDS, shuffle=True, random_state=__SEED+5)
font = {'size'   : 14}
matplotlib.rc('font', **font)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")