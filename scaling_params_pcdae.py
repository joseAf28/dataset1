import numpy as np
import yaml
import torch 
import torch.nn as nn
import math 
import h5py
import pickle
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep
import src.inference_pcdae as inference
import src.train_pcdae as train



def procedure_one(config, device, condition):
    
    seed, hidden_dim = condition
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config['model']['hidden_dim'] = hidden_dim
    
    pcdae = train.train_model(seed, device, config)
    
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_test_scaled, y_test_scaled = data_dict['test']
    scaler_X, scaler_Y = data_dict['scalers']
    
    batch_size = 128

    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    sigma_min = float(config['model']['sigma_min'])
    sigma_max = float(config['model']['sigma_max'])
    
    T1_T, T1_K = config['inference']['T1_T'], config['inference']['T1_K']
    T2_T, T2_K = config['inference']['T2_T'], config['inference']['T2_K']
    
    noise_schedule1 = np.geomspace(sigma_max, sigma_min, T1_T)
    steps_vec1 = np.linspace(T1_K, T1_K, T1_T, dtype=int)
    
    noise_schedule2 = np.geomspace(sigma_max, sigma_min, T2_T)
    steps_vec2 = np.linspace(T2_K, T2_K, T2_T, dtype=int)
    
    test_init_loss = 0.0
    test_refine_loss1 = 0.0
    test_refine_loss2 = 0.0
    
    
    for i, (x, y) in enumerate(test_dataloader):
        
        y_init = torch.randn_like(y)
        y_refined1 = inference.inference_PCDAE_base(pcdae, x, y_init, noise_schedule1, \
                                steps_vec=steps_vec1, step_size=0.01, eps_conv=1e-4, eps_clip=5e-1)
            
        y_refined2 = inference.inference_PCDAE_base(pcdae, x, y_init, noise_schedule2, \
                                steps_vec=steps_vec2, step_size=0.01, eps_conv=1e-4, eps_clip=5e-1)
        
        loss_init = nn.MSELoss()(y_init, y)
        loss_refined1 = nn.MSELoss()(y_refined1, y)
        loss_refined2 = nn.MSELoss()(y_refined2, y)
        
        test_init_loss += loss_init.item()
        test_refine_loss1 += loss_refined1.item()
        test_refine_loss2 += loss_refined2.item()
    
    
    test_init_loss /= len(test_dataloader)
    test_refine_loss1 /= len(test_dataloader)
    test_refine_loss2 /= len(test_dataloader)
    
    
    print("hidden, seed: ", hidden_dim, seed)
        
    return (seed, hidden_dim, test_init_loss, test_refine_loss1, test_refine_loss2)


if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_pcdae.yaml"
    output_file = "results/scaling_pcdaeV2.h5"
    
    seed_vec = np.arange(1, 400, 40, dtype=int)
    hidden_sizes = [25, 35, 47, 58, 68, 85, 104, 118, 132]
    
    # seed_vec = [1, 20]
    # hidden_sizes = [85, 104]
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The file 'config.yaml' was not found.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit()
    
    
    condition_vec = []
    for hidden in hidden_sizes:
        for seed in seed_vec:
            condition_vec.append([seed, hidden])
    
        
    results = Parallel(n_jobs=-1, backend="loky")(delayed(procedure_one)(config, device, condition) for condition in condition_vec)
    seed_vec, hidden_vec, loss_init_vec, loss_refine1_vec, loss_refine2_vec = zip(*results)
    
    pickled_config = pickle.dumps(config)
    
    fileh5 = h5py.File(output_file, 'w')
    group = fileh5.require_group("results")
    group.create_dataset("seed", data=seed_vec)
    group.create_dataset("hidden", data=hidden_vec)
    group.create_dataset("loss_init", data=loss_init_vec)
    group.create_dataset("loss_refine1", data=loss_refine1_vec)
    group.create_dataset("loss_refine2", data=loss_refine2_vec)
    group.create_dataset("config", data=np.void(pickled_config))
    fileh5.close()