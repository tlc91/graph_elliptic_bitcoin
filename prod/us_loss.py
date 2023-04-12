import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss helpers
"""

def _dnu_weight(dnu):
    # TODO: modify this, should be roughly mean of sqrt(dnu) of trainset
    return torch.sqrt(dnu) / 20.

def _log_loss(output, target, min_ret):
    log_output = torch.log(min_ret + output)
    log_target = torch.log(min_ret + target)
    
    return log_output, log_target

""" Loss functions
"""

def mse_loss(output, target, parameters=None):
    loss = (output - target) ** 2
    return loss.mean()


def msle_loss(output, target, dnu):
    min_ret = (1/dnu).reshape(-1,1)
    
    output, target = _log_loss(output, target, min_ret)
    
    loss = (output - target) ** 2
    
    return loss.mean()

def weighted_msle_loss(output, target, dnu):
    min_ret = (1/dnu).reshape(-1,1)
    weight = _dnu_weight(dnu) 
    
    output, target = _log_loss(output, target, min_ret)
    
    loss = (output - target) ** 2
    loss *= weight
    
    return loss.mean()

