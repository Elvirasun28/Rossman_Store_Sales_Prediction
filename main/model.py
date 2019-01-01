from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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



def xgb(train,k):
    print('>> The XGBoosting Model Selected ....................................\n')
    train = train.reset_index()
    fea = train.columns.drop(['Open', 'Sales', 'PromoInterval', 'Date'])
    train_fea = train[fea]
    train_fea = train_fea.fillna(0)
    train_y = train.Sales

    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.09, gamma=0, subsample=0.90,
                               colsample_bytree=1, max_depth=7,random_state=SEED)

    train_errs = []
    train_rmsepes = []
    val_errs = []
    val_rmsepes = []
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    print('The {} folds cross validation Begins ......................\n'.format(k))
    for train_idx, val_idx in kf.split(train_fea, train_y):
        X_train, X_val = train_fea.iloc[train_idx], train_fea.iloc[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        ## Because it is a tree-based algorithm, do not need  the feature scaling and build the model
        xgb_model.fit(X_train,y_train)
        train_error, train_rmsepe, val_error, val_rmsepe = err_metric(X_train, y_train, X_val, y_val, xgb_model)

        train_errs.append(train_error)
        train_rmsepes.append(train_rmsepe)
        val_errs.append(val_error)
        val_rmsepes.append(val_rmsepe)

        # generate report
    print('mean(train_error): {0} |RMSEPE(train): {1} |mean(val_error): {2}|RMSEPE(val): {3}'.
          format(round(np.mean(train_errs), 4), round(np.mean(train_rmsepes), 4),
                 round(np.mean(val_errs), 4), round(np.mean(val_rmsepes), 4)))
    #743|0.11031


    # save the model to disk
    xgb_model_final = xgb_model.fit(train_fea,train_y)
    filename = 'model/finalized_xgb_model.sav'
    pickle.dump(xgb_model_final, open(filename, 'wb'))

    return xgb_model_final

def lightgb_model(train,k):
    print('>> The LightingGMB Model Selected ....................................\n')
    train = train.reset_index()
    fea = train.columns.drop(['level_0','Open', 'Sales', 'PromoInterval', 'Date'])
    train_fea = train[fea]
    train_fea = train_fea.fillna(0)
    train_y = train.Sales

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.3,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'bagging_freq': 1,
        'verbose': 1
    }

    train_errs = []
    train_rmsepes = []
    val_errs = []
    val_rmsepes = []
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    print('The {} folds cross validation Begins ......................\n'.format(k))
    for train_idx, val_idx in kf.split(train_fea, train_y):
        X_train, X_val = train_fea.iloc[train_idx], train_fea.iloc[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        ## Because it is a tree-based algorithm, do not need  the feature scaling and build the model
        d_train = lgb.Dataset(X_train, label=y_train)
        lightgbm_model = lgb.train(params, d_train, 100)
        train_error, train_rmsepe, val_error, val_rmsepe = err_metric(X_train, y_train, X_val, y_val, lightgbm_model)

        train_errs.append(train_error)
        train_rmsepes.append(train_rmsepe)
        val_errs.append(val_error)
        val_rmsepes.append(val_rmsepe)

        # generate report
    print('mean(train_error): {0} |RMSEPE(train): {1} |mean(val_error): {2}|RMSEPE(val): {3}'.
          format(round(np.mean(train_errs), 4), round(np.mean(train_rmsepes), 4),
                 round(np.mean(val_errs), 4), round(np.mean(val_rmsepes), 4)))

    # mean(train_error): 553.2285 |RMSEPE(train): 0.0938 |mean(val_error): 559.4839|RMSEPE(val): 0.0971

    # save the model to disk
    d_train = lgb.Dataset(train_fea, label=train_y)
    lightgbm_model_final = lgb.train(params, d_train, 150)
    filename = 'model/finalized_lightgbm_model.sav'
    pickle.dump(lightgbm_model_final, open(filename, 'wb'))

    return lightgbm_model_final









