import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import math 
from torch.utils.data import Dataset, DataLoader

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep



def train_model(seed, device, config):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    
    X_scaled, y_scaled = data_dict['train']
    X_val_scaled, y_val_scaled = data_dict['val']
    X_test_scaled, y_test_scaled = data_dict['test']
    
    batch_size = config['training']['batch_size']

    dataset = ds.LTPDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ds.LTPDataset(X_val_scaled, y_val_scaled)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ###* define the PCDAE architecture
    x_dim = config['model']['x_dim']
    y_dim = config['model']['y_dim']
    hidden_dim = config['model']['hidden_dim']
    
    sigma_min = float(config['model']['sigma_min'])
    sigma_max = float(config['model']['sigma_max'])
    
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    
    pcdae = models.PCDAE(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim).to(device)
    
    ###* train the model
    
    lr = float(config['training']['lr'])
    num_epochs_max = config['training']['n_epochs_max']
    num_epochs_min = config['training']['n_epochs_min']
    noise_kernel = config['training']['noise_kernel']
    
    optimizer = optim.Adam(pcdae.parameters(), lr=lr)
    stopper_pcdae = models.EarlyStopping(patience=20, min_delta=1e-4, min_epochs=num_epochs_min)

    for epoch in range(1, num_epochs_max+1):
        pcdae.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            u = torch.rand(x.size(0), 1)
            noise_level = torch.exp(u * (log_max - log_min) + log_min).to(device)
            
            if noise_kernel == "VE":
                y_noisy = y + torch.randn_like(y) * noise_level
            elif noise_kernel == "VP":
                y_noisy = (1.0 - noise_level**2).sqrt() * y + torch.randn_like(y) * noise_level
            
            recon, _ = pcdae(x, y_noisy, noise_level)
            truth = torch.cat([x,y], dim=-1)
            
            loss = nn.MSELoss()(recon, truth)
            
            loss.backward()
            optimizer.step()
            
        
        pcdae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_dataloader:
                
                xv = xv.to(device)
                yv = yv.to(device)
                
                u = torch.rand(xv.size(0), 1)
                noise_level = torch.exp(u * (log_max - log_min) + log_min).to(device)
                
                if noise_kernel == "VE":
                    y_noisy = yv + torch.randn_like(yv) * noise_level
                elif noise_kernel == "VP":
                    y_noisy = (1.0 - noise_level**2).sqrt() * yv + torch.randn_like(yv) * noise_level
                
                recon, _ = pcdae(xv, y_noisy, noise_level)
                truth = torch.cat([xv,yv], dim=-1)
                
                val_loss += nn.MSELoss()(recon, truth).item()
                
            val_loss /= len(val_dataloader)
            
        
        if epoch % 400 == 0:
            print("Epoch: ", epoch, " Val_loss: ", val_loss)
        
        if stopper_pcdae.step(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("seed: ", seed, " done")
    return pcdae
