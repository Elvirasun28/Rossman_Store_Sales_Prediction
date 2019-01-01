import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main.featureEngr import *
from main.timeseries_pred import *
from main.model import *
import pickle



if __name__ == '__main__':
    ## load dataset
    rossman = pd.read_csv('data/train.csv',header = 0,encoding='utf-8',low_memory=False)
    rossman_test = pd.read_csv('data/test.csv',header = 0,encoding='utf-8',low_memory=False)
    store = pd.read_csv('data/store.csv',header = 0, encoding='utf-8',low_memory=False)

    ## data preparation
    rossman_all = feaEngr(rossman,store,train=True)
    rossman_test_all = feaEngr(rossman_test,store,train=False)

    ## store the prepared dataset
    rossman_all.to_csv('data/train_adjusted.csv')
    rossman_test_all.to_csv('data/test_adjusted.csv')

    ## distribution analysis
    print('The correlation with Sales:.......\n')
    print(rossman_all.corr()['Sales'])
    ''' We found sales would be zero every 7 days - Every Sunday expect the first day 
        so we will choose the data from 2013-01-07 for each store'''

    lr_model = linearreg(rossman_all,k=5)
    xgb_model = xgb(rossman_all,k = 5)
    lightdbm_model = lightgb_model(rossman_all,k = 5)

    ## load in model
    lightdbm_model = pickle.load(open('model/finalized_lightgbm_model.sav', 'rb'))
    xgb_model =  pickle.load(open('model/finalized_xgb_model.sav', 'rb'))
    lr_model =  pickle.load(open('model/finalized_linearregmodel.sav', 'rb'))

    ## load test_customer_adjusted dataset
    test_customer_adjusted = pd.read_csv('data/test_customer_adjusted.csv')
    test_customer_adjusted = test_customer_adjusted.drop(['Unnamed: 0','Date.1','Store.1'],axis = 1)
    ## predict the test dataset
    fea = test_customer_adjusted.columns.drop(['Id','Open','PromoInterval','Date'])
    test_fea = test_customer_adjusted[fea]
    test_pred = lightdbm_model.predict(test_fea)
    test_customer_adjusted['Sales'] = test_pred
    submission_sample = pd.read_csv('data/sample_submission.csv')
    submission = pd.merge(submission_sample, test_customer_adjusted[['Id','Sales']], how='left', on=['Id'])
    submission = submission.drop('Sales_x',axis = 1)
    submission.columns = ['Id','Sales']
    submission = submission.fillna(0)
    submission =submission.sort_values('Id')
    submission.to_csv('data/submission.csv')





