from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import lightgbm as lgb
import pickle
from main.errorMetric import *
SEED = 1


def linearreg(train,k):
    print('>> The Linear Regression Model Selected ....................................\n')
    train = train.reset_index()
    fea = train.columns.drop(['index','level_0','Open','Sales','PromoInterval','Date'])
    train_fea = train[fea]
    train_fea = train_fea.fillna(0)
    train_y = train.Sales
    lr = LinearRegression()

    train_errs = []
    train_rmsepes = []
    val_errs = []
    val_rmsepes = []
    kf = KFold(n_splits=k,shuffle=True,random_state=SEED)
    print('The {} folds cross validation Begins ......................\n'.format(k))
    for train_idx, val_idx in kf.split(train_fea,train_y):
        X_train, X_val = train_fea.iloc[train_idx],train_fea.iloc[val_idx]
        y_train, y_val = train_y[train_idx],train_y[val_idx]
        ## do the feature scaling and build the model
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        lr.fit(X_train,y_train)

        train_error, train_rmsepe,val_error,val_rmsepe = err_metric(X_train, y_train, X_val, y_val, lr)

        train_errs.append(train_error)
        train_rmsepes.append(train_rmsepe)
        val_errs.append(val_error)
        val_rmsepes.append(val_rmsepe)
        # generate report
    print('mean(train_error): {0} |RMSEPE(train): {1} |mean(val_error): {2}|RMSEPE(val): {3}'.
          format(round(np.mean(train_errs), 4),round(np.mean(train_rmsepes), 4),
                 round(np.mean(val_errs), 4),round(np.mean(val_rmsepes), 4)))
    ## 1137| 0.2045
    # save the model to disk
    train_fea = sc.fit_transform(train_fea)
    lr_final = lr.fit(train_fea,train_y)
    filename = 'model/finalized_linearregmodel.sav'
    pickle.dump(lr_final, open(filename, 'wb'))
    return lr_final



def xgb(train,num_round):
    print('>> The XGBoosting Model Selected ....................................\n')
    train = train.reset_index()
    fea = train.columns.drop(['index','Open', 'Sales', 'PromoInterval', 'Date'])
    train_fea = train[fea]
    train_y = train.Sales
    param = {'nthread': -1,
             'max_depth': 7,
             'eta': 0.02,
             'silent': 1,
             'objective': 'reg:linear',
             'colsample_bytree': 0.7,
             'subsample': 0.7}
    num_round = 2000
    X_train,X_val, y_train, y_val = train_test_split(train_fea,train_y,test_size= 0.2,random_state=SEED)
    dtrain = xgb.DMatrix(X_train, label=np.log(y_train))
    dtest = xgb.DMatrix(X_val)
    evallist = [(dtrain, 'train')]
    xgb_model = xgb.train(param, dtrain, num_round,evallist)
    ## Because it is a tree-based algorithm, do not need  the feature scaling and build the model
    train_pred = xgb_model.predict(dtrain)
    test_pred = xgb_model.predict(dtest)
    train_error, train_rmsepe, val_error, val_rmsepe = err_metric(train_pred, np.log(y_train), test_pred, np.log(y_val))
    # generate report
    print('mean(train_error): {0} |RMSEPE(train): {1} |mean(val_error): {2}|RMSEPE(val): {3}'.
          format(train_error,train_rmsepe,val_error, val_rmsepe))
    #mean(train_error): 6.19802319650124 |RMSEPE(train): 2.853937112772821 |mean(val_error): 0.42165806864090005|RMSEPE(val): 0.04744090366880039

    # save the model to disk
    filename = 'model/finalized_xgb_model.sav'
    pickle.dump(xgb_model, open(filename, 'wb'))

    return xgb_model

def lightgb_model(train,num_round):
    print('>> The LightingGMB Model Selected ....................................\n')
    train = train.reset_index()
    fea = train.columns.drop(['level_0','Open', 'Sales', 'PromoInterval', 'Date'])
    train_fea = train[fea]
    train_fea = train_fea.fillna(0)
    train_y = train.Sales

    param = {'nthread': -1,
             'max_depth': 7,
             'eta': 0.02,
             'silent': 1,
             'objective': 'regression',
             'colsample_bytree': 0.7,
             'subsample': 0.7}


    X_train, X_val, y_train, y_val = train_test_split(train_fea, train_y, test_size=0.2, random_state=SEED)
    dtrain = lgb.Dataset(X_train, label=np.log(y_train))
    dtest = lgb.Dataset(X_val)
    evallist = [(dtrain, 'train')]
    lgb_model = lgb.train(param, dtrain, num_round)
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_val)
    train_error, train_rmsepe, val_error, val_rmsepe = err_metric(train_pred, np.log(y_train), test_pred, np.log(y_val))

        # generate report
    print('mean(train_error): {0} |RMSEPE(train): {1} |mean(val_error): {2}|RMSEPE(val): {3}'.
          format(train_error, train_rmsepe,val_error, val_rmsepe))

    # mean(train_error): 553.2285 |RMSEPE(train): 0.0938 |mean(val_error): 559.4839|RMSEPE(val): 0.0971

    # save the model to disk
    filename = 'model/finalized_lightgbm_model.sav'
    pickle.dump(lgb_model, open(filename, 'wb'))

    return lgb_model


