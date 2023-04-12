import time
import numpy as np
import pandas as pd
from prod.nn_utils import remove_nans_from_array

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def main_nn_fit(train_loader, validation_loader, model, optimizer, config, lr_scheduler=None):
    """ Loop over n_epochs iterations to fit neural network to training data.
    An early stop condition is added if mean of 5 consecutive losses starts increasing.
    """
    n_epochs = config['n_epochs']
    loss_function = config['loss_function']
    epoch_count = 0
    
    train_losses, val_duds = [], []
    all_curve_params = []
    
    for ix_epoch in range(n_epochs):
        
        train_loss = train_model_loop(train_loader, model, loss_function, optimizer=optimizer)
        
        if lr_scheduler is not None:
            # updating learning rate in case of decay
            lr_scheduler.step()
        
        val_dud, curve_params = validation_loop(validation_loader, model, loss_function)
        all_curve_params.append(curve_params)
        
        train_losses.append(train_loss)
        val_duds.append(val_dud)
        
        epoch_count += 1
        # early stopping condition
        if (len(train_losses) > config['min_epoch']):
            tracked_loss = np.array([_l.detach().numpy() for _l in val_duds])
            if np.mean(tracked_loss[-10:-5]) < np.mean(tracked_loss[-5:-1]):
                #print(f'Early stopping at epoch {ix_epoch} and (train_loss,val_dud) {train_loss}, {val_dud}')
                break
    
    train_losses_arr = np.array([_l.detach().numpy() for _l in train_losses])
    val_duds_arr = np.array([_l.detach().numpy() for _l in val_duds])
    
    return train_losses_arr, val_duds_arr, all_curve_params, epoch_count


def train_model_loop(data_loader, model, loss_function, optimizer):
    """Single epoch training function"""
    
    num_samples = len(data_loader.sampler)
    total_loss = 0
    model.train()
    
    for next_batch in data_loader:
        X = next_batch['features']
        dx = next_batch['dx']
        y = next_batch['labels']
        batch_size = X.shape[0]
        
        parameters = (next_batch['dnu'], next_batch['npts'])
        
        output = model(X, dx)
        loss = loss_function(output, y, parameters)
    
        optimizer.zero_grad() 
        loss.backward() # computes gradients of all torch.tensors used to compute loss
        optimizer.step() # updates parameters based on error=gradients

        total_loss += loss * batch_size
        
    avg_loss = total_loss / num_samples
    
    return avg_loss


def _dud_calculation_per_cohort(output, target, parameters):
    dnu = parameters[0]
    
    # get mask t o get number of valid days in cohort
    mask = torch.logical_not(torch.isnan(target)) 
    number_of_days_per_cohort = mask.sum(axis=1)
    
    # sum in cohort direction, we're looking for a mean dud per cohort
    dud = (dnu.reshape(-1,1) * (output - target)).nansum(axis=1).abs()
    dud = dud / mask.sum(axis=1)
    
    return dud


def validation_loop(data_loader, model, loss_function):
    
    num_samples = len(data_loader.sampler)
    curve_params = torch.tensor([])
    num_batches = len(data_loader)
    total_loss, val_dud = 0, 0
    model.eval()
    
    with torch.no_grad():
        for next_batch in data_loader:
            X = next_batch['features']
            dx = next_batch['dx']
            y = next_batch['labels']
            batch_size = X.shape[0]
            parameters = (next_batch['dnu'], next_batch['npts'])
            
            output, curve_parameters = model(X, dx, return_param=True)
            
            val_dud += _dud_calculation_per_cohort(output, y, parameters).sum()
            
            curve_params = torch.cat((curve_params, torch.cat(curve_parameters, axis=1)), 0)
            
            loss = loss_function(output, y, parameters) 
            total_loss += loss * batch_size
        
    avg_loss = total_loss / num_samples
    val_dud = val_dud / num_samples
    
    return val_dud, curve_params


def _get_model_curve_df(forecast_df, ypred, dx, cohort_idx):
    dx = remove_nans_from_array(dx.detach().numpy().ravel()).astype(int)
    ypred = remove_nans_from_array(ypred.detach().numpy().ravel())
    forecast_df.loc[cohort_idx,dx] = ypred
    
def forecast_loop(data_loader, model, forecast_domain):
    min_dx, max_dx = forecast_domain.dx.min(), forecast_domain.dx.max()
    cohort_dates = forecast_domain.cohort_date.unique()
    
    curve_params = torch.tensor([])
    forecast_df = pd.DataFrame(columns=np.arange(min_dx,max_dx))
    
    model.eval()
    with torch.no_grad():
        for cohort_idx, next_batch in enumerate(data_loader):
            X = next_batch['features']
            dx = next_batch['dx']
            
            ypred, curve_parameters = model(X, dx, return_param=True)
            curve_params = torch.cat((curve_params, torch.cat(curve_parameters, axis=1)), 0)
            
            _get_model_curve_df(forecast_df, ypred, dx, cohort_dates[cohort_idx])
            
    return forecast_df, curve_params





