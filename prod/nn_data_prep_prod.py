import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
    
class ParamDataset(Dataset):
    
    def __init__(self, dataset, features_list):
        
        self.features = dataset[features_list].values
        self.dnu = dataset['cohort_size'].values
        self.npts = dataset['npts'].values
        self.dx = dataset['dx'].values
        self.labels = dataset['retention'].values
        self.features_list = features_list
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        features = self.features[idx]
        dnu = self.dnu[idx]
        npts = self.npts[idx]
        dx = self.dx[idx]
        labels = self.labels[idx]
        return dict(
            features=torch.tensor(features),
            dnu=torch.tensor(dnu),
            npts=torch.tensor(npts),
            dx=torch.tensor(dx),
            labels=torch.tensor(labels).to(torch.float64)
        )

def get_loaders(df, val_df, forecast_df, config, 
                 shuffle_dataset = True, random_seed = 32):

    features_list, batch_size = config['features_list'], config['batch_size']
    
    train_dataset = ParamDataset(df, features_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ParamDataset(val_df, features_list)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    forecast_dataset = ParamDataset(forecast_df, features_list)
    # batch size of forecast loader = 1 to simplify filling in final model_curve df
    forecast_loader = DataLoader(forecast_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, forecast_loader