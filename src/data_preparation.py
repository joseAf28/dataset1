import numpy as np
import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def prepare_data_model(seed, data_path, ratio_test_val_train):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ###* Prepare data
    with open(data_path, 'r') as f:
        data = f.readlines()


    for i in range(len(data)):
        data[i] = data[i].split()
        data[i] = [float(x) for x in data[i]]
        
    data = np.array(data)

    X_train, X_temp, y_train, y_temp = train_test_split(data[:, 0:3], data[:, 3:], test_size=ratio_test_val_train, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=seed)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_Y.fit_transform(y_train)

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_Y.transform(y_val)

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_Y.transform(y_test)


    return {'scalers': (scaler_X, scaler_Y), 
            'train': (X_scaled, y_scaled),
            'val': (X_val_scaled, y_val_scaled),
            'test': (X_test_scaled, y_test_scaled)
            }