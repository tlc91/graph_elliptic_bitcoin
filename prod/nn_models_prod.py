import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Prob_Pareto_BB(nn.Module):
    ''' Pareto - BB model as a torch.nn layer.
    Returns probability of PBB model'''
    def __init__(self):
        super().__init__()

    def forward(self, dx, c, s, g):
        prob = torch.where(s==1, self._singularity_pbb(dx, c, s, g), self._pbb(dx, c, s, g))
        return prob
    
    def _pbb(self, dx, c, s, g):
        return c / (s-1) * g**s * (1/(g+dx)**(s-1) - 1/(g+dx+1)**(s-1))#c * g / (s-1)  * ((1+dx/g)**(1-s) - (1+(dx+1)/g)**(1-s))
    
    def _singularity_pbb(self, dx, c, s, g):
        return c * g * (torch.log(1/(g+dx)) - torch.log(1/(g+dx+1)))

    
class Log_Prob_Pareto_BB(nn.Module):
    ''' Log Prob to more efficiently transform PBB.
    !! Issue: pure log will not work as we cannot take log of y target
    (many 0s).'''
    def __init__(self):
        super().__init__()

    def forward(self, dx, c, s, g):
        prob = torch.where(s==1, self._singularity_pbb(dx, c, s, g), self._pbb(dx, c, s, g))
        return prob
    
    def _pbb(self, dx, c, s, g):
        return torch.log(c) - torch.log(s-1) + s * torch.log(g) + torch.log((g+dx)**(1-s) - (g+dx+1)**(1-s))
    
    def _singularity_pbb(self, dx, c, s, g):
        return torch.log(c) + torch.log(g) +  torch.log(torch.log(1/(g+dx)) - torch.log(1/(g+dx+1)))

    
class Prob_Pareto_BB_MS(nn.Module):
    ''' Pareto - BB model with minimum of survivors (MS) as a torch.nn layer.
    Returns probability of PBBMS model'''
    def __init__(self):
        super().__init__()

    def forward(self, t, pi, c, s, g):
        prob = c * (pi + (1 - pi) / (s-1) * g**s * (1/(g+t)**(s-1) - 1/(g+t+1)**(s-1)))
        return prob
    

class ParetoBBNet(nn.Module):
    '''Main NN model definition.
    Initialized with number of feature, dropout coefficient, layer size factor and model name.
    Forward pass outputs by default the predicted ypred, but can also return the parameters.
    '''
    def __init__(self, config):
        super().__init__()
        
        n_features = len(config['features_list'])
        l1, l2 = config['l1'], config['l2']
        dropout_coef = config['dropout_coef']
        model_name = config['model_name']
        
        self.first_layer = nn.Linear(n_features, l1)  
        self.hidden_layer = nn.Linear(l1, l2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_coef)
        self.softplus = nn.Softplus(beta=0.5)
        self.exp = torch.exp
        
        self.model_name = model_name
        
        if self.model_name=='pbb':
            self.model = Prob_Pareto_BB()
            n_output_parameters = 3
        elif self.model_name=='pbbms':
            self.model = Prob_Pareto_BB_MS()
            n_output_parameters = 4
        else:
            raise Exception("Model unknown. ", self.model)
           
        self.last_layer = nn.Linear(l2, n_output_parameters)
        
        torch.nn.init.xavier_uniform_(self.first_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.last_layer.weight)
        
    
    def forward(self, x, dx, return_param=False):
        mask = torch.isnan(dx)
        
        x = self.relu(self.dropout(self.first_layer(x)))
        x = self.relu(self.dropout(self.hidden_layer(x)))
        x = self.last_layer(x)
       
        if self.model_name=='pbb':  
            c = torch.sigmoid(x[:,0]).reshape(-1,1)
            s = self.softplus(x[:,1]).reshape(-1,1)
            g = self.softplus(x[:,2]).reshape(-1,1)
            
            # fill dx with 0s to avoid nan propagation  
            # and enforce y_pred to 0s where
            y_pred = self.model(dx.where(torch.logical_not(mask), torch.tensor(0.0)), c, s, g)
            y_pred = y_pred.masked_fill(mask, torch.nan)
            
            if return_param:
                return y_pred, (c,s,g)
            else:
                return y_pred
            
             
        elif self.model_name=='pbbms':
            pi = torch.sigmoid(x[:,0]).reshape(-1,1)
            c = torch.sigmoid(x[:,1]).reshape(-1,1)
            s = self.exp(-x[:,2]).reshape(-1,1)
            g = self.exp(-x[:,3]).reshape(-1,1)

            y_pred = self.model(dx[mask], pi, c, s, g)

            if return_param:
                return y_pred, (pi,c,s,g)
            else:
                return y_pred
            
        