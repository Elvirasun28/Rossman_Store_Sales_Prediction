from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import  numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pyramid.arima import auto_arima
import fbprophet
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima_model import ARIMA

def checkStat(data):
    # Determine rolling statistics
    rolmean = data.Customers.rolling(window=6).mean()
    rolstd = data.Customers.rolling(window=6).std()

    # Plot rolling statistics: (also first 100 values)
    c = list(data.Customers.values[7:])
    plt.figure(figsize=(12,6))
    orig = plt.plot(c, color='blue', label='Original')
    mean = plt.plot(rolmean[7:].astype('int'), color='red', label='Rolling Mean')
    std = plt.plot(rolstd[7:].astype('int'), color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    ## p-value < 0.05, which means that it is stationary
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data.Customers, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def timePred(data, period):
    ## check the stationary or not
    data_sorted = data.sort_index()
    # checkStat(data_sorted_1)
    data_sorted.index = [datetime.strptime(d, '%Y-%m-%d').date() for d in data_sorted.index]
    ## pred the future data
    stepwise_model = auto_arima(data_sorted.Customers, start_p=0, start_q=1,
                                max_p=0, max_q=3,start_P=0,seasonal=True,
                                d=1, D=1, trace=True, error_action='ignore',
                                suppress_warnings=True,stepwise=True)
    print('The AIC for the optimal ARIMA model: ', stepwise_model.aic())
    ## predict the customer values in test_set

    stepwise_model.fit(data_sorted.Customers)
    test_pred = stepwise_model.predict(n_periods= period)

    return list(test_pred)


def testCustPred(train,test):
    train_data = pd.DataFrame(train[['Store','Customers']].values, index=train.Date.values)
    train_data.columns = ['Store','Customers']
    testD =test.Date.values[:41]
    cust = pd.DataFrame(columns=['Store','Customers'])
    for st in set(test.Store):
        print('Predict Customer Amount For store {}\n'.format(st))
        train_data_store = train_data[train_data.Store == st]
        custMax = pd.DataFrame(timePred(train_data_store,len(testD)),columns=['Customers'],index=testD)
        custMax['Store'] = [st] * custMax.shape[0]
        cust = pd.concat([cust,custMax],sort=False)

    cust.Store = cust.Store.astype('int32')
    cust.Customers = cust.Cusomters.astype('int32')
    cust = cust[42:]
    test = pd.merge(test, cust, on='Store')
    test.to_csv('data/test_customer_adjusted.csv')
    return test






def lastMonthCustoemr(train,test):
    train_data = pd.DataFrame(train[['Store', 'Customers']].values, index=train.Date.values)
    train_data.columns = ['Store', 'Customers']
    cust_pd = pd.DataFrame(columns=['Store','Customers'])
    for st in set(test.Store):
        print('Predict Customer Amount For store {}\n'.format(st))
        dateD = test[test.Store == st].Date.values
        dateD = [datetime.strptime(d, '%Y-%m-%d').date() for d in dateD]
        if [train_data[train_data.Store == st].loc['%s' % str(d + relativedelta(years=-1))] for d in dateD ]:

            value = train_data[train_data.Store == st][-len(dateD):].Customers.values.copy()
        customer = pd.DataFrame(value, columns=['Customers'],index= dateD)
        customer['Store'] = st
        cust_pd = pd.concat([cust_pd, customer], sort=False)
    cust_pd = cust_pd.reset_index()
    cust_pd.columns = ['Date','Store','Customers']
    cust_pd.Store = cust_pd.Store.astype('int32')
    result = pd.concat([test, cust_pd], axis=1, sort=False)
    result.to_csv('data/test_customer_adjusted.csv')
    return result
