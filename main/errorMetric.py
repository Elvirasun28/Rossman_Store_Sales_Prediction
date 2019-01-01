import numpy as np
from sklearn.metrics import mean_squared_error

def train_err(X_train, y_train,model):
    pred = model.predict(X_train)
    y_train = y_train.values
    pred = [0 if p < 1000 else int(p) for p in pred]
    mse = mean_squared_error(y_train, pred)
    rmse = np.sqrt(mse)
    rmsepe = np.sqrt(np.mean([((pred[i] - y_train[i]) / y_train[i])**2 for i in range(len(y_train)) if y_train[i] != 0]))
    return rmse,rmsepe

def val_err(X_val, y_val, model):
    pred = model.predict(X_val)
    y_val = y_val.values
    pred = [0 if p < 1000 else p for p in pred]
    mse = mean_squared_error(y_val,pred)
    rmse = np.sqrt(mse)
    rmsepe = np.sqrt(
        np.mean([((pred[i] - y_val[i]) / y_val[i]) ** 2 for i in range(len(y_val)) if y_val[i] != 0]))
    return rmse,rmsepe

def err_metric(X_train,y_train,X_val, y_val, model):
    train_error,train_rmsepe = train_err(X_train,y_train, model)
    val_error,val_rmsepe = val_err(X_val,y_val,model)
    return train_error,train_rmsepe,val_error,val_rmsepe