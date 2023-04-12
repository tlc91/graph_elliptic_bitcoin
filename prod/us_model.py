import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnsplashedNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        n_features = len(config['features_list'])
        l1, l2 = config['l1'], config['l2']
        dropout_coef = config['dropout_coef']
        
        self.first_layer = nn.Linear(n_features, l1)  
        self.hidden_layer = nn.Linear(l1, l2)
        
        self.last_layer = nn.Linear(l2, 3)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_coef)
        self.sigmoid = nn.Sigmoid()
        self.pbb_model = Prob_Pareto_BB()
      
    
    def forward(self, x, dx=None):
        
        x = self.relu1(self.dropout(self.first_layer(x)))
        x = self.relu2(self.dropout(self.hidden_layer(x)))
        x = self.last_layer(x)
        
        if dx is not None:
            c = torch.sigmoid(x[:,0])
            s = torch.exp(-x[:,1])
            g = torch.exp(-x[:,2])
            y_pred = self.pbb_model(dx, c, s, g)
            return y_pred
        else:
            return self.sigmoid(x)
        
        
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