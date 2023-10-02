import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class FocalLoss(_Loss):
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.0, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        
    def forward(self, preds, trues):
        ba_ce = F.cross_entropy(preds, trues, weight=self.weight, reduction=self.reduction, label_smoothing=self.label_smoothing)
        ln_pt = F.cross_entropy(preds, trues, reduction=self.reduction)
        pt = torch.exp(-ln_pt)
        return (((1 - pt) ** self.gamma) * ba_ce).mean()

    
class JaccardLoss(_Loss):
    def __init__(self, epsilon=1e-6):
        super(JaccardLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = torch.sum(y_pred * y_true, 0)
        fp = torch.sum(y_pred * (1 - y_true), 0)
        fn = torch.sum((1 - y_pred) * y_true, 0)
        jacc = ((tp + self.epsilon) / (tp + fp + fn + self.epsilon)).mean()
        return (1 - jacc)

    
class DiceLoss(_Loss):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = torch.sum(y_pred * y_true, 0)
        fp = torch.sum(y_pred * (1 - y_true), 0)
        fn = torch.sum((1 - y_pred) * y_true, 0)
        dice = ((2 * tp + self.epsilon)/ (2 * tp + fp + fn + self.epsilon)).mean()
        return (1 - dice)
    
    
class TverskyLoss(_Loss):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = torch.sum(y_pred * y_true, 0)
        fp = torch.sum(y_pred * (1 - y_true), 0)
        fn = torch.sum((1 - y_pred) * y_true, 0)
        tver = ((tp + self.epsilon)/ (tp + (self.alpha * fp) + (self.beta * fn) + self.epsilon)).mean()
        return (1 - tver)
    
    
class FocalTverskyLoss(_Loss):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = torch.sum(y_pred * y_true, 0)
        fp = torch.sum(y_pred * (1 - y_true), 0)
        fn = torch.sum((1 - y_pred) * y_true, 0)
        tver = ((tp + self.epsilon)/ (tp + (self.alpha * fp) + (self.beta * fn) + self.epsilon)).mean()
        return (1 - tver) ** 1/self.gamma
    
    
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            