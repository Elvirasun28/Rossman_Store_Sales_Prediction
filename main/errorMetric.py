import numpy as np
from sklearn.metrics import mean_squared_error

def train_err(train_pred, y_train):
    y_train = y_train.values
    mse = mean_squared_error(y_train, train_pred)
    rmse = np.sqrt(mse)
    rmsepe = np.sqrt(np.mean([((train_pred[i] - y_train[i]) / y_train[i])**2 for i in range(len(y_train)) if y_train[i] != 0]))
    return rmse,rmsepe

def val_err(val_pred, y_val):
    y_val = y_val.values
    mse = mean_squared_error(y_val,val_pred)
    rmse = np.sqrt(mse)
    rmsepe = np.sqrt(
        np.mean([((val_pred[i] - y_val[i]) / y_val[i]) ** 2 for i in range(len(y_val)) if y_val[i] != 0]))
    return rmse,rmsepe

def err_metric(train_pred,y_train,val_pred, y_val):
    train_error,train_rmsepe = train_err(train_pred,y_train)
    val_error,val_rmsepe = val_err(val_pred,y_val)
    return train_error,train_rmsepe,val_error,val_rmsepe
