from datetime import datetime
import pandas as pd
import numpy as np
from time import mktime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def timeSplit(date):
    dt = datetime.strptime(date,'%Y-%m-%d')
    yr = dt.year
    month = dt.month
    day = dt.day
    return yr, month, day


def feaEngr(record, store,train = True):
    ## merge the sale dataset with the store information dataset
    data_all = pd.merge(record, store, on='Store')
    print('The shape for dataset: ', data_all.shape)
    ## distribution analysis
    print('The correlation with Sales:.......\n')
    print(data_all.corr()['Sales'])
    ''' We found sales would be zero every 7 days - Every Sunday expect the first day 
        so we will choose the data from 2013-01-07 for each store'''
    ## ignore the sales == 0
    data_all_copy = data_all.copy()
    data_all_copy_sorted = data_all_copy.sort_values(['Store', 'Date'])
    data_all_copy_sorted.Open = data_all_copy_sorted.Open.fillna(0)
    if train:
        data_all_adjusted = data_all_copy_sorted[data_all_copy_sorted.Date >= '2013-01-06'][
            data_all_copy_sorted.Sales != 0]
    else:
        data_all_adjusted = data_all_copy_sorted[data_all_copy_sorted.Open != 0.0]


    ## split the date to year, month, day
    dateMat = np.squeeze([timeSplit(d) for d in data_all_adjusted.Date.values])
    data_all_adjusted['year'],data_all_adjusted['month'],data_all_adjusted['day'] = dateMat[:,0],dateMat[:,1],dateMat[:,2]
    ## change date to timstamp
    data_all_adjusted['timestamp'] = [int(mktime(datetime.strptime(d, "%Y-%m-%d").timetuple())) for d in data_all_adjusted.Date]

    ## change the character label to number
    label = {'a':1, 'b':2, 'c':3, 'd':4}
    data_all_adjusted.StateHoliday.replace(label,inplace=True)
    data_all_adjusted.Assortment.replace(label,inplace=True)
    data_all_adjusted.StoreType.replace(label,inplace=True)

    ## deal with missing value
    print('The missing value for CompetitionDistance: ', sum(np.isnan(data_all_adjusted.CompetitionDistance)))
    print('The percentage is: ', np.round(sum(np.isnan(data_all_adjusted.CompetitionDistance))/data_all.shape[0],5))
    # since there are not too many missing values in this columns
    data_all_adjusted.CompetitionDistance = data_all_adjusted.CompetitionDistance.fillna(0)
    # since N.A. means that there are no competitors near the store, I fill 0 to the N.A. values
    print('The missing value for CompetitionOpensince Month: ',sum(np.isnan(data_all_adjusted.CompetitionOpenSinceMonth)))
    print('The percentage is: ', np.round(sum(np.isnan(data_all_adjusted.CompetitionOpenSinceMonth)) / data_all.shape[0], 5))
    # since there are quite many missing values but less than 50%, for training models, I fill the meidan values
    data_all_adjusted.CompetitionOpenSinceMonth = data_all_adjusted.CompetitionOpenSinceMonth.fillna(
                                                np.nanmean(data_all_adjusted.CompetitionOpenSinceMonth)).astype('int')
    # same as competitionopen since year
    data_all_adjusted.CompetitionOpenSinceYear = data_all_adjusted.CompetitionOpenSinceYear.fillna(
                                                np.nanmean(data_all_adjusted.CompetitionOpenSinceYear)).astype('int')
    ## there are also many other features, but too many missing values, will not be used as features in the model training.

    ## transform the date type
    if train:
        data_all_adjusted[['DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']] = data_all[['DayOfWeek', 'Sales', 'Customers', \
                                                                                    'Open', 'Promo', 'SchoolHoliday']].astype('int')
        data_all_adjusted.StateHoliday = data_all_adjusted.StateHoliday.astype('int')
    else:
        data_all_adjusted.DayOfWeek = data_all_adjusted.DayOfWeek.astype('int')
        data_all_adjusted.Open = data_all_adjusted.Open.astype('int')
        data_all_adjusted.Promo = data_all_adjusted.Promo.astype('int')
        data_all_adjusted.SchoolHoliday = data_all_adjusted.SchoolHoliday.astype('int')
        data_all_adjusted.StateHoliday = data_all_adjusted.StateHoliday.astype('int')


    ## since 1/log(CompetitionDistance) has high correlation with Sales)
    data_all_adjusted['CompetitionDistance_log'] = 1 / np.log(data_all_adjusted['CompetitionDistance'])
    ## put CompetitionOpenSinceMonth,CompetitionOpenSinceYear into one features
    ohlb = OneHotEncoder(sparse=False)
    CompetitionOpenSinceMonth_onhot = ohlb.fit_transform(
        data_all_adjusted.CompetitionOpenSinceMonth.values.reshape((-1, 1)))
    CompetitionOpenSinceYear_onhot = ohlb.fit_transform(
        data_all_adjusted.CompetitionOpenSinceYear.values.reshape((-1, 1)))
    competOpenDateTime = np.concatenate([CompetitionOpenSinceMonth_onhot, CompetitionOpenSinceYear_onhot], axis=1)
    for i in range(competOpenDateTime.shape[1]):
        data_all_adjusted['competOpenDateTime_%d' % (i + 1)] = competOpenDateTime[:, i]

    ## Put Promo,Promo into one feature
    Promo_onhot = ohlb.fit_transform(data_all_adjusted.Promo.values.reshape((-1, 1)))
    Promo2_onhot = ohlb.fit_transform(data_all_adjusted.Promo2.values.reshape((-1, 1)))
    Promo_all = np.concatenate([Promo_onhot, Promo2_onhot], axis=1)
    for i in range(Promo_all.shape[1]):
        data_all_adjusted['PromoAll_%d' % (i + 1)] = Promo_all[:, i]

    ## PromoInterval
    data_all_adjusted.PromoInterval = data_all_adjusted.PromoInterval.fillna(0)
    mapper = {'Jan':1,'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sept':9,
              'Oct':10,'Nov':11, 'Dec':12}
    PromoInterval_split = [item.split(',')if item != 0 else 0 for item in data_all_adjusted.PromoInterval ]
    data_all_adjusted['PromoInterval_num'] = [len(item) if item != 0 else 0 for item in PromoInterval_split]
    PromoInterval_dates = []
    for value in PromoInterval_split:
        if value == 0:
            data = [0]*12
        else:
            data = [0]*12
            data[mapper[value[0]]-1] = 1
            data[mapper[value[1]]-1] = 1
            data[mapper[value[2]]-1] = 1
            data[mapper[value[3]]-1] = 1
        PromoInterval_dates.append(data)
    PromoInterval_dates = np.array(PromoInterval_dates)

    for i in range(12):
        data_all_adjusted['PromoInterval_%d' % (i + 1)] = PromoInterval_dates[:,i]

    print('The dataset preparation finished .......................................')
    return data_all_adjusted