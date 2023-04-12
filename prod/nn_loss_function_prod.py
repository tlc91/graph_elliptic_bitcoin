import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

""" Loss and masks helpers
"""

def _sample_size_weight_fn(npts):
    # weight based on # of points in cohort. Penalizes if npts < 50.
    return 1 - 0.8*np.exp(-npts/50.)


def _dnu_weight(dnu):
    # TODO: modify this, should be roughly mean of sqrt(dnu) of trainset
    return torch.sqrt(dnu) / 20.

def _mask_output_target(output, target):
    # Number of valid points to average loss
    loss_mask = torch.logical_not(torch.isnan(target))
    
    output = output.masked_fill(torch.isnan(output), 0)
    target = target.masked_fill(torch.isnan(target), 0)
    
    return output, target, loss_mask
    
def _log_loss(output, target, min_ret):
    log_output = torch.log(min_ret + output)
    log_target = torch.log(min_ret + target)
    
    return log_output, log_target

def _reduced_masked_loss(loss, loss_mask, reduce='cohort'):
    if reduce=='cohort':
        
        loss = loss.nansum(axis=1) / loss_mask.sum(axis=1)
        loss = loss.mean()
    elif reduce=='sum':
        loss = loss.masked_fill(torch.logical_not(loss_mask), 0).sum()
    elif reduce=='mean':
        loss = loss.masked_fill(torch.logical_not(loss_mask), 0).sum() / loss_mask.sum()
    else:
        raise "Unknown loss reduction type : " + reduce
    return loss


""" Loss functions
"""

def mse_loss(output, target, parameters=None):
    output, target, loss_mask = _mask_output_target(output, target)
    loss = (output - target) ** 2
    
    return _reduced_masked_loss(loss, loss_mask)


def msle_loss(output, target, parameters):
    dnu = parameters[0]
    min_ret = (1/dnu).reshape(-1,1)
    
    output, target, loss_mask = _mask_output_target(output, target)
    output, target = _log_loss(output, target, min_ret)
    
    loss = (output - target) ** 2
    
    return _reduced_masked_loss(loss, loss_mask)


def weighted_msle_loss(output, target, parameters):
    dnu = parameters[0].reshape(-1,1)
    min_ret = (1/dnu).reshape(-1,1)
    weight = _dnu_weight(dnu) 
    
    output, target, loss_mask = _mask_output_target(output, target)
    output, target = _log_loss(output, target, min_ret)
    
    loss = (output - target) ** 2
    loss *= weight
    
    return _reduced_masked_loss(loss, loss_mask)


def triangle_weighted_mse_loss(output, target, parameters):
    dnu, npts = parameters
    dnu = dnu.reshape(-1,1)
    min_ret = (1/dnu).reshape(-1,1)
    npts = npts.reshape(-1,1)
    
    sample_size_weight = _sample_size_weight_fn(npts)
    dnu_weight = _dnu_weight(dnu) 
    weight = sample_size_weight * dnu_weight
    
    output, target, loss_mask = _mask_output_target(output, target)
    
    loss = (output - target) ** 2
    loss *= weight
    
    return _reduced_masked_loss(loss, loss_mask)


def triangle_weighted_msle_loss(output, target, parameters):
    dnu, npts = parameters
    dnu = dnu.reshape(-1,1)
    min_ret = (1/dnu).reshape(-1,1)
    npts = npts.reshape(-1,1)
    
    sample_size_weight = _sample_size_weight_fn(npts)
    dnu_weight = _dnu_weight(dnu) 
    weight = sample_size_weight * dnu_weight
    
    output, target, loss_mask = _mask_output_target(output, target)
    output, target = _log_loss(output, target, min_ret)
    
    loss = (output - target) ** 2
    loss *= weight
    
    return _reduced_masked_loss(loss, loss_mask)

