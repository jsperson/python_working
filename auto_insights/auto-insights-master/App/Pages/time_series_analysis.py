#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:48:06 2019

@author: rachelnana
"""

import pandas as pd
import numpy as np
import pickle
import itertools
import configparser
import math
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import detrend
from Pages.Module_2_DataPrep import stripNonAlphanumeric

# pickle load
def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK,"rb") as f:
        temp_item = pickle.load(f)
    return temp_item

# pickle save
def pickle_save(name, item):
    PIK = str(name)+ ".pickle"
    with open(PIK,"wb") as f:
        pickle.dump(item, f)

# find optimal resampling frequency
def optimalResampleFreq(ts_data):
    # find optimal resampling frequency

    # cycle through various resampling methods
    for frq in ['H', 'D', 'W', 'MS', 'QS', 'YS']:
        time_agg_here = ts_data.resample(frq).mean()
        cnt = time_agg_here.isna().sum().values
        try:
            test = pd.infer_freq(time_agg_here.dropna(axis=0).index)
        except ValueError:
            test = None
        # if 0 nulls or if <5% is null, choose this resampling period
        if (cnt == 0) & (test != None):
            resample_period = frq
            break
        if (frq == 'YS') & (test == None):
            resample_period = frq

    return resample_period


# test if we need to drop any records to get a better resampling frequency
def optimalDropAndResampleParams(ts_data, tc):
    best_drop_first_num = 0
    best_drop_last_num = 0
    best_frq = 'YS'
    if optimalResampleFreq(ts_data) == 'H':
        return 0, 0, 'H'
    else:
        max_num_to_drop = min(20, round(.2*len(ts_data)))+1
        target_frq = optimalResampleFreq(ts_data[max_num_to_drop:len(ts_data)-max_num_to_drop])
        for drop_last in np.arange(max_num_to_drop):
            for drop_first in np.arange(max_num_to_drop):
                ts_subset = ts_data[drop_first:len(ts_data)-drop_last]
                resample_pd = optimalResampleFreq(ts_subset)
                if (best_frq == 'YS') & (resample_pd != 'YS'):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'QS') & (resample_pd != 'YS') & (resample_pd != 'QS'):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'MS') & (resample_pd != 'YS') & (resample_pd != 'QS') & (resample_pd != 'MS'):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'MS') & (resample_pd != 'YS') & (resample_pd != 'QS') & (resample_pd != 'MS'):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'W') & ((resample_pd == 'D') | (resample_pd == 'H')):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'D') & (resample_pd == 'H'):
                    best_drop_first_num = drop_first
                    best_drop_last_num = drop_last
                    best_frq = resample_pd
                if (best_frq == 'H') | (best_frq == target_frq):
                    break
            if (best_frq == 'H') | (best_frq == target_frq):
                break
        return best_drop_first_num, best_drop_last_num, best_frq


# return empty control stats df
def emptyControlStats(time_agg, tc, cc):
    # the trend, seasonality, and residual values will all be null
    iterables = [[tc], [cc]]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature'])
    control_stats = pd.DataFrame(np.nan, index=idx, columns=['period', 'drop_first', 'drop_last', 'nobs', 'decr', 'incr', 'blw_mn', 'abv_mn', 'blw_lcl', 'abv_ucl'])

    control_stats.loc[(tc, cc), 'period'] = 'YS'
    control_stats.loc[(tc, cc), 'drop_first'] = 0
    control_stats.loc[(tc, cc), 'drop_last'] = 0
    control_stats.loc[(tc, cc), 'nobs'] = len(time_agg)
    control_stats.loc[(tc, cc), 'decr'] = 0
    control_stats.loc[(tc, cc), 'incr'] = 0
    control_stats.loc[(tc, cc), 'blw_mn'] = 0
    control_stats.loc[(tc, cc), 'abv_mn'] = 0
    control_stats.loc[(tc, cc), 'blw_lcl'] = 0
    control_stats.loc[(tc, cc), 'abv_ucl'] = 0

    return control_stats

# return empty auto-correlation function
def emptyACF(time_agg, tc, cc):
    iterables = [[tc], [cc], np.arange(1)]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'lag'])
    acf_here = pd.DataFrame(np.nan, index=idx, columns=['Auto-Correlation'])

    return acf_here

# return empty time trends df
def emptyTimeTrends(time_agg, tc, cc):
    # the trend, seasonality, and residual values will all be null
    iterables = [[tc], [cc], ['mul', 'add', 'None'], np.arange(time_agg.shape[0])]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'type', 'period'])
    trends_here = pd.DataFrame(np.nan, index=idx, columns=['Datetime', 'Original', 'Trend', 'Seasonality', 'Residual'])

    trends_here.loc[(tc, cc, 'add'), 'Datetime'] = time_agg.index
    trends_here.loc[(tc, cc, 'add'), 'Original'] = time_agg[cc].values
    trends_here.loc[(tc, cc, 'mul'), 'Datetime'] = time_agg.index
    trends_here.loc[(tc, cc, 'mul'), 'Original'] = time_agg[cc].values

    return trends_here

# return empty time trends df
def emptyTimeForecats(time_agg, tc, cc):
    # the trend, seasonality, and residual values will all be null
    iterables = [[tc], [cc], np.arange(1)]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'datetime'])
    forecast_here = pd.DataFrame(np.nan, index=idx, columns=['Datetime', 'Forecast'])

    forecast_here.loc[(tc, cc), 'Datetime'] = time_agg.index[(len(time_agg)-1)]
    forecast_here.loc[(tc, cc), 'Forecast'] = time_agg.values[(len(time_agg)-1)]

    return forecast_here


# get the process control statistics
def processControlStats(time_agg, tc, cc, resample_period, drop_first, drop_last):
    # starting from the most recent observation, the value has been
    # increasing or decreasing for x consecutive periods
    decr_cnt = 0
    incr_cnt = 0
    for idx in reversed(time_agg.index):
        val = time_agg.loc[idx,cc]
        if idx != time_agg[-1:].index:
            if val > prev_val:
                if incr_cnt == 0:
                    decr_cnt += 1
                else:
                    break
            if val < prev_val:
                if decr_cnt == 0:
                    incr_cnt += 1
                else:
                    break
        prev_val = val

    # starting from the most recent observation, the value has been
    # above or below the mean for x consecutive periods
    mean = np.mean(time_agg.values)
    abv_mn = 0
    blw_mn = 0
    for idx in reversed(time_agg.index):
        val = time_agg.loc[idx,cc]
        if val < mean:
            if abv_mn == 0:
                blw_mn += 1
            else:
                break
        if val > mean:
            if blw_mn == 0:
                abv_mn += 1
            else:
                break

    # how many observations are more than 3 standard deviations away from the mean
    std = np.std(time_agg.values, ddof=1)
    lcl = mean-3*std
    ucl = mean+3*std
    blw_lcl = 0
    abv_ucl = 0
    if lcl != ucl:
        for idx in reversed(time_agg.index):
            val = time_agg.loc[idx,cc]
            if val <= lcl:
                if abv_ucl == 0:
                    blw_lcl += 1
                else:
                    break
            if val >= ucl:
                if blw_lcl == 0:
                    abv_ucl += 1
                else:
                    break
            if (lcl < val) & (val < ucl):
                break

    # is the most recent value more than 3 standard deviations away from the mean
    curr_abv_ucl = np.squeeze(time_agg[-1:].values >= ucl)
    curr_blw_lcl = np.squeeze(time_agg[-1:].values <= lcl)

    # store results in Multi-Index df
    iterables = [[tc], [cc]]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature'])
    control_stats = pd.DataFrame(np.nan, index=idx, columns=['period', 'drop_first', 'drop_last', 'nobs', 'decr', 'incr', 'blw_mn', 'abv_mn', 'blw_lcl', 'abv_ucl'])
    control_stats.loc[(tc, cc), 'period'] = resample_period
    control_stats.loc[(tc, cc), 'drop_first'] = drop_first
    control_stats.loc[(tc, cc), 'drop_last'] = drop_last
    control_stats.loc[(tc, cc), 'nobs'] = len(time_agg)
    control_stats.loc[(tc, cc), 'decr'] = decr_cnt
    control_stats.loc[(tc, cc), 'incr'] = incr_cnt
    control_stats.loc[(tc, cc), 'blw_mn'] = blw_mn
    control_stats.loc[(tc, cc), 'abv_mn'] = abv_mn
    control_stats.loc[(tc, cc), 'blw_lcl'] = blw_lcl
    control_stats.loc[(tc, cc), 'abv_ucl'] = abv_ucl

    return control_stats


# get auto-correlation function
def getACF(time_agg, tc, cc):
    try:
        acf_array = acf(time_agg)
        acf_array = acf_array[1:]
        size = acf_array.size
        acf_status = 'pass'
    except:
        print('Auto-Correlation function for', tc,'vs.',cc,'failed.')
        size = 1
        acf_status = 'fail'

    iterables = [[tc], [cc], np.arange(size)]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'lag'])
    acf_here = pd.DataFrame(np.nan, index=idx, columns=['Auto-Correlation'])

    if acf_status == 'pass':
        acf_here.loc[(tc, cc), 'Auto-Correlation'] = acf_array

    return acf_here


# perform additive/multiplicative trend seasonality decomposition
def seasDecomposition(time_agg, tc, cc, add_or_mul='add'):
    if add_or_mul != 'mul':
        add_or_mul='add'

    # decompose into trend, seasonality, and residual
    if add_or_mul == 'add':
        try:
            decomposition = seasonal_decompose(time_agg)
            trend = decomposition.trend.values
            seasonal = decomposition.seasonal.values
            resid = decomposition.resid.values
            decomposition_status = 'pass'
        except:
            print('Additive seasonal decomposition for',tc,'vs.',cc,'failed.')
            decomposition_status = 'fail'
    if add_or_mul == 'mul':
        try:
            decomposition = seasonal_decompose(time_agg, model='multiplicative')
            trend = decomposition.trend.values
            seasonal = decomposition.seasonal.values
            resid = decomposition.resid.values
            decomposition_status = 'pass'
        except:
            print('Multiplicative seasonal decomposition for',tc,'vs.',cc,'failed.')
            decomposition_status = 'fail'

    # create a multi-index dataframe containing the original trend, seasonal, and residual results
    iterables = [[tc], [cc], [add_or_mul], np.arange(time_agg.shape[0])]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'type', 'period'])
    trends_here = pd.DataFrame(np.nan, index=idx, columns=['Datetime', 'Original', 'Trend', 'Seasonality', 'Residual'])

    trends_here.loc[(tc, cc, add_or_mul), 'Datetime'] = time_agg.index
    trends_here.loc[(tc, cc, add_or_mul), 'Original'] = time_agg[cc].values
    if decomposition_status == 'pass':
        trends_here.loc[(tc, cc, add_or_mul), 'Trend'] = trend
        trends_here.loc[(tc, cc, add_or_mul), 'Seasonality'] = seasonal
        trends_here.loc[(tc, cc, add_or_mul), 'Residual'] = resid

    return decomposition_status, trends_here

# return time trends multi-index array with null seasonality component
def noSeasonality(time_agg, tc, cc, level):
    trend = level
    seasonal = np.empty(time_agg.shape)
    seasonal.fill(np.nan)
    resid = np.squeeze(time_agg.values) - trend

    # create a multi-index dataframe containing the original trend, seasonal, and residual results
    iterables = [[tc], [cc], ['None'], np.arange(time_agg.shape[0])]
    idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'type', 'period'])
    trends_here = pd.DataFrame(np.nan, index=idx, columns=['Datetime', 'Original', 'Trend', 'Seasonality', 'Residual'])

    trends_here.loc[(tc, cc, 'None'), 'Datetime'] = time_agg.index
    trends_here.loc[(tc, cc, 'None'), 'Original'] = time_agg[cc].values
    trends_here.loc[(tc, cc, 'None'), 'Trend'] = trend
    trends_here.loc[(tc, cc, 'None'), 'Seasonality'] = seasonal
    trends_here.loc[(tc, cc, 'None'), 'Residual'] = resid

    return trends_here

# get number of periods that make up a 'season'
def numSeasonalPeriods(df_column):
    # get number of periods between peaks
    try:
        seas_max = max(df_column.values)
        peaks = (df_column == seas_max).mul(1)
        peaks2 = peaks.reset_index()
        first_peak = peaks2[peaks2.iloc[:,1] == 1].reset_index().loc[0,'index']
        second_peak = peaks2[peaks2.iloc[:,1] == 1].reset_index().loc[1,'index']
        num_seas_periods = (second_peak - first_peak)
    except:
        num_seas_periods = 2

    # If decompose method did not detect any seasonal trends, set number of seasonal periods to 2.
    # Otherwise, exponential smoothing will fail.
    if num_seas_periods < 2:
        num_seas_periods = 2

    return num_seas_periods

# return sum of squared ACF
def sumOfSquareACF(df_column):
    if df_column.dropna(axis=0).shape[0] < 2:
        return math.inf
    acf_array = acf(df_column.dropna(axis=0))
    acf_array = acf_array[1:]
    acf_sq = np.square(acf_array).sum()

    return acf_sq

# exponential smoothing
def expSmoothForecast(time_agg, tc, cc, resample_period, return_level=False, seas_param=None, seas_period_param=None):
    # subset of time_agg
    train_size = round(len(time_agg)*.8)
    time_agg_train = time_agg[:train_size]
    time_agg_test = time_agg[train_size:]

    # check seasonal parameters
    if seas_param == None:
        seas_period_param = None

    # iterate over these parameters
    if np.nanmin(time_agg) > 0:
        trends = [None, 'add', 'mul']
    else:
        trends = [None, 'add']
    damps = [False, True]

    # find best values for trend and damped
    min_sse = math.inf
    best_trend_param = None
    best_damped_param = False
    for trend_param in trends:
        for damped_param in damps:
            if (trend_param != None) | ((trend_param == None) & (damped_param == False)):
                try:
                    ets_stl_train_here = ExponentialSmoothing(time_agg_train.values, trend=trend_param, damped=damped_param, seasonal=seas_param, seasonal_periods=seas_period_param).fit()
                    ets_stl_train_fcast = ets_stl_train_here.forecast(time_agg_test.size)
                    diff = np.squeeze(time_agg_test.values,axis=1) - ets_stl_train_fcast
                    sse = (diff ** 2).sum()
                    if sse < min_sse:
                        min_sse = sse
                        best_trend_param = trend_param
                        best_damped_param = damped_param
                except:
                    min_sse = min_sse
                    best_trend_param = best_trend_param
                    best_damped_param = best_damped_param

    # run Exponential Smoothing with best parameters, on full history
    try:
        ets_stl = ExponentialSmoothing(time_agg.values, trend=best_trend_param, damped=best_damped_param, seasonal=seas_param, seasonal_periods=seas_period_param).fit()
        length = int(np.ceil(len(time_agg)/4))
        ets_stl1 = ets_stl.forecast(length)
        level = ets_stl.level
        exp_smoothing_status = 'pass'
    except:
        exp_smoothing_status = 'fail'
        if seas_param == 'add':
            print('Exponential Smoothing for',tc,'vs.',cc,'with additive seasonality failed.')
        elif seas_param == 'mul':
            print('Exponential Smoothing for',tc,'vs.',cc,'with multiplicative seasonality failed.')
        else:
            print('Exponential Smoothing for',tc,'vs.',cc,'with no seasonality failed.')

    # create a multi-index dataframe containing the forecast results
    if exp_smoothing_status == 'fail':
        forecast_here = emptyTimeForecats(time_agg, tc, cc)
        level = np.full(len(time_agg), np.nan)
    else:
        iterables = [[tc], [cc], np.arange(ets_stl1.shape[0]+1)]
        idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature', 'datetime'])
        forecast_here = pd.DataFrame(np.nan, index=idx, columns=['Datetime', 'Forecast'])
        forecast_here.loc[(tc, cc), 'Datetime'] = pd.date_range(start=time_agg.index[(len(time_agg)-1)], periods=(length+1), freq=resample_period)
        forecast_here.loc[(tc, cc), 'Forecast'] = np.append(time_agg.values[(len(time_agg)-1)], ets_stl1)

    if return_level == True:
        return forecast_here, str(best_trend_param), str(best_damped_param), min_sse, level
    else:
        return forecast_here, str(best_trend_param), str(best_damped_param), min_sse


# time series pre-processing: check for data sparsity, find optimal resampling frequency,
# and produce process control statistics
def timeSeriesPreProcessing(clean_df_time, tc, cc):
    print('Time-Series Preprocessing for',tc,'vs.',cc);
    time_ser = clean_df_time.loc[:,[tc, cc]].dropna(axis=0).set_index(tc);
    time_ser = time_ser.sort_values(by=tc);

    # resample by year and check for > 500 nulls
    if time_ser.resample('YS').mean().isna().sum().values > 500:
        print('Data too sparse. Skipping',tc,'vs.',cc)
        time_agg = time_ser.resample('YS').mean().sort_values(by=tc)
        resample_period = None

        # create a multi-index dataframe containing the process control stats
        control_stats_here = emptyControlStats(time_agg, tc, cc)
    else:
        drop_first, drop_last, resample_period = optimalDropAndResampleParams(time_ser, tc)
        if drop_first > 0:
            print('Dropping first',drop_first,'records.')
            time_ser = time_ser[drop_first:]
        if drop_last > 0:
            print('Dropping first',drop_last,'records.')
            time_ser = time_ser[:len(time_ser)-drop_last]
        print('Resampling by', resample_period)
        # get avg over each time period
        time_agg = time_ser.resample(resample_period).mean().sort_values(by=tc)

        # process control stats
        control_stats_here = processControlStats(time_agg, tc, cc, resample_period, drop_first, drop_last)

    return control_stats_here


# run time series analysis betwen time column, tc, and feature column, cc
def runTimeSeries(clean_df_time, resample_period, drop_first, drop_last, tc, cc):
    print('Time-Series Analysis of',tc,'vs.',cc);

    # use optimal drop and resample parameters to take periodic averages across time
    time_ser = clean_df_time.loc[:,[tc, cc]].dropna(axis=0).set_index(tc);
    time_ser = time_ser.sort_values(by=tc);
    if drop_first > 0:
        time_ser = time_ser[drop_first:]
    if drop_last > 0:
        time_ser = time_ser[:len(time_ser)-drop_last]
    time_agg = time_ser.resample(resample_period).mean().sort_values(by=tc)

    # resample by year and check for > 500 nulls
    if resample_period == None:
        # create a multi-index dataframe containing the auto-correlation function
        acf_here = emptyACF(time_agg, tc, cc)
        # create a multi-index dataframe containing the original trend, seasonal, and residual results
        trends_here = emptyTimeTrends(time_agg, tc, cc)
        # create a multi-index dataframe containing the forecast results
        forecast_here = emptyTimeForecats(time_agg, tc, cc)

        # empty best model df
        iterables = [[tc], [cc]]
        idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature'])
        best_model_here = pd.DataFrame(np.nan, index=idx, columns=['Trend', 'Damped', 'Seasonality', 'Seasonal Periods'])

    else:
        # get auto-correlation function
        acf_here = getACF(time_agg, tc, cc)

        # get additive and multiplicative decompositions
        add_status, additive_trends_here = seasDecomposition(time_agg, tc, cc, 'add')
        trends_here = additive_trends_here
        mul_status, multiplicative_trends_here = seasDecomposition(time_agg, tc, cc, 'mul')
        trends_here = trends_here.append(multiplicative_trends_here)

        # residual ACFs
        acf_sos = sumOfSquareACF(time_agg);
        if add_status == 'pass':
            add_acf = sumOfSquareACF(trends_here.loc[(tc, cc, 'add'), 'Residual']);
        if mul_status == 'pass':
            mul_acf = sumOfSquareACF(trends_here.loc[(tc, cc, 'mul'), 'Residual']);

        # best seasonality model
        if (add_status == 'fail') & (mul_status == 'fail'):
            best_seas_model = 'None'
        elif (add_status == 'fail') & (mul_status == 'pass'):
            if mul_acf < acf_sos:
                best_seas_model = 'mul'
            else:
                best_seas_model = 'None'
        elif (add_status == 'pass') & (mul_status == 'fail'):
            if add_acf < acf_sos:
                best_seas_model = 'add'
            else:
                best_seas_model = 'None'
        else:
            if (add_acf <= mul_acf) & (add_acf < acf_sos):
                best_seas_model = 'add'
            elif (mul_acf <= add_acf) & (mul_acf < acf_sos):
                best_seas_model = 'mul'
            else:
                best_seas_model = 'None'

        print('Best Seasonal Model:', best_seas_model)

        # exponential smoothing
        if best_seas_model == 'add':
            # find number of periods in a 'season'
            best_seas_periods = numSeasonalPeriods(trends_here.loc[(tc, cc, 'add'), 'Seasonality'])
            print('Number of seasonal periods (additive model):',best_seas_periods)
            forecast_here, best_trend_param, best_damped_param, min_sse = expSmoothForecast(time_agg, tc, cc, resample_period, False, 'add', best_seas_periods)
        elif best_seas_model == 'mul':
            # find number of periods in a 'season'
            best_seas_periods = numSeasonalPeriods(trends_here.loc[(tc, cc, 'mul'), 'Seasonality'])
            print('Number of seasonal periods (multiplicative model):',best_seas_periods)
            forecast_here, best_trend_param, best_damped_param, min_sse = expSmoothForecast(time_agg, tc, cc, resample_period, False, 'mul', best_seas_periods)
        else:
            best_seas_model = 'None'
            best_seas_periods = 'None'
            forecast_here, best_trend_param, best_damped_param, min_sse, none_level = expSmoothForecast(time_agg, tc, cc, resample_period, True)
            none_trends_here = noSeasonality(time_agg, tc, cc, none_level)
            trends_here = trends_here.append(none_trends_here)

        iterables = [[tc], [cc]]
        idx = pd.MultiIndex.from_product(iterables, names=['time variable', 'feature'])
        best_model_here = pd.DataFrame(np.nan, index=idx, columns=['Trend', 'Damped', 'Seasonality', 'Seasonal Periods'])
        best_model_here.loc[(tc, cc), 'Trend'] = best_trend_param
        best_model_here.loc[(tc, cc), 'Damped'] = best_damped_param
        best_model_here.loc[(tc, cc), 'Seasonality'] = best_seas_model
        best_model_here.loc[(tc, cc), 'Seasonal Periods'] = best_seas_periods

    return acf_here, trends_here, forecast_here, best_model_here


# run time series analysis in one-off fashion for time_col vs. feature and append to storeded data in ts_runs, ts_trends, ts_forecast
def runOneOffTimeSeries(time_col, feature):
    config = configparser.ConfigParser()
    config.read('./ml_box.ini')
    runtime_settings = config['RUNTIME']
    labelField = runtime_settings['label_field']
    dash_data_path = runtime_settings['dash_data_path']

    #clean labelField
    labelField = stripNonAlphanumeric([labelField])[0]

    # load time series data
    clean_df_time_path = dash_data_path + 'clean_df_time'
    clean_df_time = pickle_load(clean_df_time_path)
    date_cols_path = dash_data_path + 'date_cols'
    date_cols = pickle_load(date_cols_path)
    non_date_cols_path = dash_data_path + 'non_date_cols'
    non_date_cols = pickle_load(non_date_cols_path)
    non_date_cols.append(labelField)

    # load previous run information
    ts_runs_path = dash_data_path + 'ts_runs'
    ts_runs = pickle_load(ts_runs_path)
    ts_control_stats_path = dash_data_path + 'ts_control_stats'
    ts_control_stats = pickle_load(ts_control_stats_path)
    ts_acf_path = dash_data_path + 'ts_acf'
    ts_acf = pickle_load(ts_acf_path)
    ts_trends_path = dash_data_path + 'ts_trends'
    ts_trends = pickle_load(ts_trends_path)
    ts_forecast_path = dash_data_path + 'ts_forecast'
    ts_forecast = pickle_load(ts_forecast_path)
    ts_best_model_path = dash_data_path + 'ts_best_model'
    ts_best_model = pickle_load(ts_best_model_path)

    if time_col == 'all':
        print('cycle over all time variables')
        time_vars = date_cols
    else:
        print('time variable:', time_col)
        time_vars = [time_col]

    if feature == 'all':
        print('cycle over all non-time features')
        non_time_vars = non_date_cols
    else:
        print('feature:', feature)
        non_time_vars = [feature]

    # all combinations of variables to run
    all_combos = list(itertools.product(time_vars,non_time_vars))

    # eliminate combinations that have already run
    try:
        ts_list = ts_runs.to_records(index=False).tolist()
        first_run = False
    except:
        ts_list = []
        first_run = True
    to_run = [x for x in all_combos if x not in ts_list]

    for tuple in to_run:
        tc = tuple[0]
        cc = tuple[1]

        # optimal drop and resample parameters
        resample_period = ts_control_stats.loc[(tc,cc),'period']
        drop_first = int(ts_control_stats.loc[(tc,cc),'drop_first'])
        drop_last = int(ts_control_stats.loc[(tc,cc),'drop_last'])

        # run time series analysis between tc and cc
        acf_here, trends_here, forecast_here, best_model_here = runTimeSeries(clean_df_time, resample_period, drop_first, drop_last, tc, cc)

        # append results to ts_runs, ts_trends, and ts_forecast
        if first_run:
            ts_acf = acf_here
            ts_trends = trends_here
            ts_forecast = forecast_here
            ts_best_model = best_model_here
            ts_runs = pd.DataFrame(data={'time_var': [tc], 'feature': [cc]})
        else:
            ts_acf = ts_acf.append(acf_here)
            ts_trends = ts_trends.append(trends_here)
            ts_forecast = ts_forecast.append(forecast_here)
            ts_best_model = ts_best_model.append(best_model_here)
            ts_runs = ts_runs.append({'time_var': tc, 'feature': cc}, ignore_index=True)

        ts_acf_path = dash_data_path + 'ts_acf'
        pickle_save(ts_acf_path, ts_acf)
        ts_runs_path = dash_data_path + 'ts_runs'
        pickle_save(ts_runs_path, ts_runs)
        ts_trends_path = dash_data_path + 'ts_trends'
        pickle_save(ts_trends_path, ts_trends)
        ts_forecast_path = dash_data_path + 'ts_forecast'
        pickle_save(ts_forecast_path, ts_forecast)
        ts_best_model_path = dash_data_path + 'ts_best_model'
        pickle_save(ts_best_model_path, ts_best_model)

    return True
