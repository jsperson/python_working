#!/usr/bin/env python
# coding: utf-8

#import datetime
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, log_loss
#from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import pandas as pd
import pickle
from sklearn import metrics
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
#from Pages.Module_2_DataPrep import returnCleanDf
from Pages.Module_1_DataIngestion import generateDf
import configparser
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import timedelta
from datetime import datetime
from Pages.time_series_analysis import runTimeSeries
from Pages.time_series_analysis import timeSeriesPreProcessing
import re
import itertools

def stripNonAlphanumeric(items): #(keeps underscores)
    items = [x.strip() for x in items] #strip whitespace on left or right
    items = [re.sub(' |-', '_', x) for x in items] #replace spaces and dashed with unnderscore
    items = [re.sub('\W|^_', '', x) for x in items] #keep only number, letters, and underscores
    items = ["_"+x if x[0].isdigit() else x for x in items] #if first character is number, prefix with "_"
    return items

def module4():
    config = configparser.ConfigParser()
    config.read('./ml_box.ini')
    runtime_settings = config['RUNTIME']
    labelField = runtime_settings['label_field']
    labelField = stripNonAlphanumeric([labelField])[0] #clean labelField
    dash_data_path = runtime_settings['dash_data_path']

    ingest_settings = config['INGEST']
    data_type = ingest_settings['datatype']

    if data_type == 'sql':
        data_source = ingest_settings['TABLE_NAME']
    else:
        data_source = ingest_settings['file_name']

    #Load best model
    def pickle_load(name):
        PIK = str(name) + ".pickle"
        with open(PIK,"rb") as f:
            temp_item = pickle.load(f)
        return temp_item

    best_model_path = dash_data_path + 'best_model_automl'
    best_model = pickle_load(best_model_path)

    X_train_path = dash_data_path + 'X_train'
    X_train = pickle_load(X_train_path)
    X_test_path = dash_data_path + 'X_test'
    X_test = pickle_load(X_test_path)
    Y_train_path = dash_data_path + 'Y_train'
    Y_train = pickle_load(Y_train_path)
    Y_test_path = dash_data_path + 'Y_test'
    Y_test = pickle_load(Y_test_path)

    def pickle_save(name, item):
        PIK = str(name)+ ".pickle"
        with open(PIK,"wb") as f:
            pickle.dump(item, f)

    Y_pred = best_model.predict(X_test)

    #ROC curve
    #calculate the fpr and tpr for all thresholds of the classification
    probs = best_model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    #feature metrics for sensitivity analysis
#    featureMeans = X_train.mean()
    featureMins = X_train.min()
    featureMaxs = X_train.max()
    featureMetrics = pd.concat([featureMins, featureMaxs], axis=1)
    featureMetrics.columns = ['min', 'max']

    #confusion matrix
    conf = confusion_matrix(y_true = Y_test, y_pred = Y_pred)

    # ## Market Basket
    mb_ignore = runtime_settings['mb_ignore']

    if mb_ignore == 'false':
        clean_df_path = dash_data_path + 'clean_df'
        norm_df_clean = pickle_load(clean_df_path)

        #drop non-categorical columns
        norm_df = norm_df_clean.select_dtypes(include=['int64', 'uint8', 'float64'])

        #one more filter - get rid of int columns that are not categorical (tenure)
        #get rid of columns that are never 'negative' (always 1 or greater)
        continuous_int_cols = list(norm_df.loc[:, (norm_df<=0).sum()==0].columns)
        print("Market basket analysis - removed columns that are never 0:", continuous_int_cols)

        norm_df.drop(continuous_int_cols, axis=1, inplace=True)

        #drop target variable
        norm_df.drop(labelField, axis=1, inplace=True)

        # drop collinear columns (ex. dropping _No phone services attributes, since PhoneService_No covers these)
        # NOTE - gives preference for columns appearing first in the data
        cols_to_drop = []
        for x in norm_df.columns:
            for y in norm_df.columns:
                corr = norm_df[x].corr(norm_df[y])
                if x != y and (corr >= 0.99):
                    if y not in cols_to_drop and x not in cols_to_drop:
                        cols_to_drop.append(y)

        norm_df.drop(labels=cols_to_drop, axis=1, inplace=True)

        def encode_units(x):
            if x < 1:
                return 0
            if x >= 1:
                return 1

        basket_sets = norm_df.applymap(encode_units)

        #If support is too high, lower to allow values in
        try:
            frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True, max_len=3)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        except ValueError:
            frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True, max_len=3)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)       

        max_lift = rules[ (rules['lift'] >= 1) &
                   (rules['conviction'] != np.inf)]
        market_basket = max_lift.sort_values(by='lift', ascending=False).head(100) #limit to top 100
        market_basket['antecedents'] = market_basket['antecedents'].map(lambda x: list(x))
        market_basket['consequents'] = market_basket['consequents'].map(lambda x: list(x))

        #Drop baskets containing the same items but reversed
        market_basket['unique'] = (market_basket['antecedents'] + market_basket['consequents']).apply(
            lambda x: ' '.join(sorted(x)))

        market_basket.drop_duplicates(subset=['unique'], keep='first', inplace=True) #drop a<->c baskets

        #Calculate top Lift and Support cells, set dummy value in new column for later market basket conditional formatting
        lift_top10_thresh = market_basket['lift'].quantile(q=0.9) #90th percentile threshold #
        market_basket['lift_highlight'] = np.where(market_basket['lift']>=lift_top10_thresh, 1, 0)
        support_top10_thresh = market_basket['support'].quantile(q=0.9) #90th percentile threshold #
        market_basket['support_highlight'] = np.where(market_basket['support']>=support_top10_thresh, 1, 0)

    #    market_basket['lift'] = market_basket['lift'].map(lambda x: '{:.2f}x'.format(x))
    #    market_basket['support'] = market_basket['support'].map(lambda x: '{:.0%}'.format(x))
        market_basket['Basket'] = (market_basket['antecedents'] + market_basket['consequents']).apply(lambda x: ', '.join(x))

        market_basket = market_basket[['Basket', 'support', 'lift', 'lift_highlight', 'support_highlight']]
        market_basket.columns = ['Basket', 'Support', 'Lift', 'lift_highlight', 'support_highlight']

        #Remove any baskets whose items are a subset of a larger basket (preference for basket specificity)
        all_baskets = list(market_basket['Basket'].unique())
        all_baskets = [set(x.split(', ')) for x in all_baskets]

        print("Removing basket subsets...")
        remove_baskets = []

        for index, row in market_basket.iterrows():
            b = row['Basket']
            b_set = set(b.split(', '))

            for a in all_baskets:
                if b_set != a and b_set.issubset(a): #if basket has superset in all_baskets, remove
                    remove_baskets.append(b)

        market_basket = market_basket[~market_basket['Basket'].isin(remove_baskets)]

        market_basket.reset_index(inplace=True, drop=True)

        # ### Get averages and sums for all other columns by these market baskets

        calc_dict = {}
        calc_dict_cols = []

        for b in market_basket.Basket.values:
            b_parsed = b.split(', ')

            basket_filtered_df = norm_df_clean.copy() #create df where all items in basket will be true
            for item in b_parsed:
                basket_filtered_df = basket_filtered_df[basket_filtered_df[item]>=1]

            for c in basket_filtered_df.columns: #find agg measures of each column
                basket_filtered_column = basket_filtered_df[c] #in basket universe
                m = basket_filtered_column.mean()
                s = basket_filtered_column.sum()
                cnt = basket_filtered_column.count()

                population_column = norm_df_clean[c] #in total universe
                p_m = population_column.mean()
                p_s = population_column.sum()
                p_cnt = population_column.count()

                basket_col_name = c+'_basket'
                pop_col_name = c+'_pop'

                if b in calc_dict.keys(): # data
                    calc_dict[b] += [s, m, cnt, p_s, p_m, p_cnt]
                else:
                    calc_dict[b] = [s, m, cnt, p_s, p_m, p_cnt]

                # columns
                for name_of_column in [basket_col_name+'_sum', basket_col_name+'_mean', basket_col_name+'_count',
                                     pop_col_name+'_sum', pop_col_name+'_mean', pop_col_name+'_count']:
                    if name_of_column not in calc_dict_cols:
                        calc_dict_cols.append(name_of_column)


        market_basket_calcs = pd.DataFrame.from_dict(calc_dict, orient='index', columns=calc_dict_cols)


        #Format calc columns
    #    calc_format = [col for col in market_basket_calcs if col.endswith('sum') or col.endswith('count') or col.endswith('mean')]
    #    market_basket_calcs[calc_format] = market_basket_calcs[calc_format].applymap(lambda x: '{:.2f}'.format(x)
    #                                              if len(str(round(x))) <= 1
    #                                              else '{:.0f}'.format(x))

        market_basket_calcs['Basket'] = market_basket_calcs.index

        #join calcs to market basket
        market_basket = market_basket.merge(market_basket_calcs, on='Basket', how='left')

        # Find top 10th percentile for each calculated field
        calc_cols = [col for col in market_basket if col.endswith('sum') or col.endswith('count') or col.endswith('mean')]
        for col in calc_cols:
            top10_thresh = market_basket[col].quantile(q=0.9) #90th percentile threshold
            market_basket[col+'_highlight'] = np.where(market_basket[col]>=top10_thresh, 1, 0)

        #Replace "_" with " = " for readability
    #    market_basket['Basket'] = market_basket['Basket'].str.replace('_', ' = ')

        market_basket_csv_path = dash_data_path + 'market_basket.csv'
        market_basket.to_csv(market_basket_csv_path, index=False)


    # ### Data summary
    norm_df_summary = generateDf()

    data_summary = norm_df_summary.head(1)
    data_summary_csv_path = dash_data_path + 'data_summary.csv'
    data_summary.to_csv(data_summary_csv_path, index=False)
    clean_df_path = dash_data_path + 'clean_df'
    data_post_transform = pickle_load(clean_df_path)
    data_summary_post_transform = data_post_transform.head(1)
    data_summary_post_transform_csv_path = dash_data_path + 'data_summary_post_transform.csv'
    data_summary_post_transform.to_csv(data_summary_post_transform_csv_path, index=False)
    data_post_transform_csv_path = dash_data_path + 'data_post_transform.csv'
    data_post_transform.to_csv(data_post_transform_csv_path, index=False)

    # ## Permutation-based feature importance

    feature_names = list(X_test.columns.values)
    perm = PermutationImportance(best_model).fit(X_test, Y_test)

    ex = eli5.explain_weights(perm, feature_names = feature_names)

    perm_feature_wt = eli5.formatters.as_dataframe.format_as_dataframe(ex)

    perm_feature_wt = dict(perm_feature_wt[['feature','weight']])

    #create perm_feature_wt for pre-dummified data (for single instance prediction tab)
    dummy_memory_path = dash_data_path + 'dummy_memory'
    dummy_memory = pickle_load(dummy_memory_path)
    dummy_memory = {x[:-3]:y for x,y in dummy_memory.items()} #remove ' = ' separator from parent

    def returnPreDummyCol(postDummyCol): #returns pre-dummification column/parent
        for dummy_parent, dummy_children in dummy_memory.items():
            if postDummyCol in dummy_children:
                return dummy_parent
        return postDummyCol

    perm_feature_wt_predummy = perm_feature_wt.copy()
    perm_feature_wt_predummy = pd.DataFrame(perm_feature_wt_predummy)
    perm_feature_wt_predummy['feature'] = perm_feature_wt_predummy['feature'].map(returnPreDummyCol)
    perm_feature_wt_predummy = perm_feature_wt_predummy[['feature']]
    perm_feature_wt_predummy.drop_duplicates(inplace = True)
    perm_feature_wt_predummy.reset_index(inplace=True, drop=True)

    #Create pre-dummified features_metrics for use in single instance prediction tab
    featureMetricsPreDummy = featureMetrics.copy()
    featureMetricsPreDummy['preDummy'] = featureMetricsPreDummy.index.map(returnPreDummyCol)
    featureMetricsPreDummy.drop_duplicates(subset=['preDummy'], inplace=True)
    #erase calculations for predummy variables
    # feature_metrics['mean'] = np.where(feature_metrics.index != feature_metrics['preDummy'], float('nan'), feature_metrics['mean'])
    featureMetricsPreDummy['min'] = np.where(featureMetricsPreDummy.index != featureMetricsPreDummy['preDummy'], float('nan'), featureMetricsPreDummy['min'])
    featureMetricsPreDummy['max'] = np.where(featureMetricsPreDummy.index != featureMetricsPreDummy['preDummy'], float('nan'), featureMetricsPreDummy['max'])

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)

    #export pre-dummified data sample for use in single instance prediction
    sample_size = 200
    if X_test.shape[0] < sample_size:
        sample_size = X_test.shape[0]
    rand_prediction = X_test.sample(n=sample_size)

    #Given set of observations, return probability of 1 using best model
    def get_pos_proba_from_x(observation):
        best_model_automl_path = dash_data_path + 'best_model_automl'
        best_model_pipeline = pickle_load(best_model_automl_path)
        probabilities = best_model_pipeline.predict_proba([observation])
        return probabilities[0][1]

    rand_prediction['pos_proba'] = rand_prediction.apply(get_pos_proba_from_x, axis=1)

    #create index sorted by pos_proba for eventual slider
    rand_prediction.sort_values(by='pos_proba', inplace=True)

    #get pre-dummified data for same set of indices
    sample_indices = rand_prediction.index

    pre_dummified_clean_df_path = dash_data_path + 'pre_dummified_clean_df'
    pre_dummy_data = pickle_load(pre_dummified_clean_df_path)

    rand_prediction_pre_dummy = pre_dummy_data.loc[sample_indices]
    rand_prediction_pre_dummy['pos_proba'] = rand_prediction['pos_proba']
    rand_prediction_pre_dummy.reset_index(inplace=True, drop=True)

    rf_feature_importances = rf_feature_importances[0:6]

    feature_names = list(X_test.columns.values)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)
    ex = eli5.explain_weights(clf, feature_names = feature_names)
    reg_feature_weight = eli5.formatters.as_dataframe.format_as_dataframe(ex)
    top_3 = reg_feature_weight.head(3)
    bottom_3 = reg_feature_weight.tail(3)

    analysis_lines = []

    first_feature_pos = top_3.iloc[[0]]['feature'].item()
    second_feature_pos = top_3.iloc[[1]]['feature'].item()
    third_feature_pos = top_3.iloc[[2]]['feature'].item()

    second_feature_ratio = top_3.iloc[[1]]['weight'].item()/top_3.iloc[[0]]['weight'].item()
    third_feature_ratio = top_3.iloc[[2]]['weight'].item()/top_3.iloc[[0]]['weight'].item()

    analysis_lines.append('For predicting %s classes of %s, the strongest predictors are: ' % (runtime_settings['pos_label'], runtime_settings['label_field']))
    analysis_lines.append('The strongest predictor is %s' % (first_feature_pos))
    analysis_lines.append('The second strongest predictor is %s with a weight %0.2f%% of %s' % (second_feature_pos, second_feature_ratio, first_feature_pos))
    analysis_lines.append('The third strongest predictor is %s with a weight %0.2f%% of %s' % (third_feature_pos, third_feature_ratio, first_feature_pos))

    first_feature_neg = bottom_3.iloc[[2]]['feature'].item()
    second_feature_neg = bottom_3.iloc[[1]]['feature'].item()
    third_feature_neg = bottom_3.iloc[[0]]['feature'].item()

    second_feature_ratio = bottom_3.iloc[[0]]['weight'].item()/bottom_3.iloc[[1]]['weight'].item()
    third_feature_ratio = bottom_3.iloc[[0]]['weight'].item()/bottom_3.iloc[[2]]['weight'].item()

#    analysis_lines.append('For predicting %s classes of %s, the strongest predictors are: ' % (runtime_settings['neg_label'], runtime_settings['label_field']))
    analysis_lines.append('The strongest predictor is %s' % (first_feature_neg))
    analysis_lines.append('The second strongest predictor is %s with a weight %0.2f%% of %s' % (second_feature_neg, second_feature_ratio, first_feature_neg))
    analysis_lines.append('The third strongest predictor is %s with a weight %0.2f%% of %s' % (third_feature_neg, third_feature_ratio, first_feature_neg))


    # Time Series Analysis
    # load cleansed data with time variables
    clean_df_time_path = dash_data_path + 'clean_df_time'
    clean_df_time = pickle_load(clean_df_time_path)

    # datetime columns
    date_cols_path = dash_data_path + 'date_cols'
    date_cols = pickle_load(date_cols_path)
    all_date_cols_path = dash_data_path + 'all_date_cols'
    all_date_cols = pickle_load(all_date_cols_path)
    # all other columns
    non_date_cols_path = dash_data_path + 'non_date_cols'
    non_date_cols = pickle_load(non_date_cols_path)
    all_non_date_cols_path = dash_data_path + 'all_non_date_cols'
    all_non_date_cols = pickle_load(all_non_date_cols_path)

    # If ts_ignore = true, skip time series analysis.
    if runtime_settings['ts_ignore'] == 'true':
        print('Ignore Time Series = TRUE. Skipping time series analysis.')
        ts_acf_path = dash_data_path + 'ts_acf'
        pickle_save(ts_acf_path, None)
        ts_runs_path = dash_data_path + 'ts_runs'
        pickle_save(ts_runs_path, None)
        ts_trends_path = dash_data_path + 'ts_trends'
        pickle_save(ts_trends_path, None)
        ts_forecast_path = dash_data_path + 'ts_forecast'
        pickle_save(ts_forecast_path, None)
        ts_best_model_path = dash_data_path + 'ts_best_model'
        pickle_save(ts_best_model_path, None)
    # If no datetime data, there is no analysis to be done. Continue on.
    elif len(date_cols) == 0:
        print('No datetime data. Skipping time series analysis.')
        ts_acf_path = dash_data_path + 'ts_acf'
        pickle_save(ts_acf_path, None)
        ts_runs_path = dash_data_path + 'ts_runs'
        pickle_save(ts_runs_path, None)
        ts_trends_path = dash_data_path + 'ts_trends'
        pickle_save(ts_trends_path, None)
        ts_forecast_path = dash_data_path + 'ts_forecast'
        pickle_save(ts_forecast_path, None)
        ts_best_model_path = dash_data_path + 'ts_best_model'
        pickle_save(ts_best_model_path, None)
    else:
        # resample and get process control stats for all time-feature pairs
        for tc in date_cols:
            for cc in non_date_cols:
                control_stats_here = timeSeriesPreProcessing(clean_df_time, tc, cc)

                if (non_date_cols.index(cc) == 0) & (date_cols.index(tc) == 0):
                    ts_control_stats = control_stats_here
                else:
                    ts_control_stats = ts_control_stats.append(control_stats_here)

            # use process control stats to select which variables to run
            decreasing = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'decr'] > 7)]['decr'].sort_values(ascending=False)
            increasing = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'incr'] > 7)]['incr'].sort_values(ascending=False)
            below_mn = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'blw_mn'] > 7)]['blw_mn'].sort_values(ascending=False)
            above_mn = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'abv_mn'] > 7)]['abv_mn'].sort_values(ascending=False)
            below_lcl = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'blw_lcl'] > 0)]['blw_lcl'].sort_values(ascending=False)
            above_ucl = ts_control_stats.loc[tc][(ts_control_stats.loc[(tc), 'abv_ucl'] > 0)]['abv_ucl'].sort_values(ascending=False)
            top_n = 2 # we will choose top_n features from each category to run against each time variable
            to_run_list_here = list(itertools.product([tc],decreasing[:top_n].index)) + list(itertools.product([tc],increasing[:top_n].index)) + list(itertools.product([tc],below_mn[:top_n].index)) + list(itertools.product([tc],above_mn[:top_n].index)) + list(itertools.product([tc],below_lcl[:top_n].index)) + list(itertools.product([tc],above_ucl[:top_n].index))
            to_run_list_here = list(set(to_run_list_here))

            if date_cols.index(tc) == 0:
                to_run_list = to_run_list_here
            else:
                to_run_list = to_run_list + to_run_list_here

        # remove duplicates - hopefully this step is unnecessary
        to_run_list = list(set(to_run_list))
        print('to run:',to_run_list)

        # save TS process control stats
        ts_control_stats_path = dash_data_path + 'ts_control_stats'
        pickle_save(ts_control_stats_path, ts_control_stats)


        # top 7 most informative features
        perm_feature_wt_df = pd.DataFrame(perm_feature_wt)
        num_features = min((perm_feature_wt_df.shape[0]-1), 6)
        top_10 = perm_feature_wt_df.sort_values('weight',ascending=False).reset_index(drop=True).loc[:num_features,'feature'].values.tolist()
        top_10.append(labelField)
        #for cc in top_10:
        #    for tc in date_cols:
        for tuple in to_run_list:
            tc = tuple[0]
            cc = tuple[1]

            # optimal drop and resample parameters
            resample_period = ts_control_stats.loc[(tc,cc),'period']
            drop_first = int(ts_control_stats.loc[(tc,cc),'drop_first'])
            drop_last = int(ts_control_stats.loc[(tc,cc),'drop_last'])

            acf_here, trends_here, forecast_here, best_model_here = runTimeSeries(clean_df_time, resample_period, drop_first, drop_last, tc, cc)

            # append results to the larger multi-index dfs
            #if (date_cols.index(tc) == 0) & (top_10.index(cc) == 0):
            if to_run_list.index(tuple) == 0:
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


    # ## Prepare for export

    x_100_test = X_test[:100]
    # print(x_100_test.shape)

    #Time to classify 100 new samples
    start = time.clock()

    best_model.predict(x_100_test)

    elapsed = time.clock() - start
    # print(elapsed)

    #Prepare various metrics for export
    model_type = (' ').join(list(best_model.named_steps.keys()))
    timestamp = datetime.now()
    params_json = best_model.get_params()
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    log_loss_score = log_loss(Y_test, Y_pred)
#    mae = mean_absolute_error(Y_test, Y_pred)
#    mse = mean_squared_error(Y_test, Y_pred)
    # roc_auc = #roc_auc already defined above
    test_item_count = Y_test.count()
    run_time = elapsed

    #More metrics
    file_name=data_source
    row_count=norm_df_summary.shape[0]
    col_count=norm_df_summary.shape[1]
    target_variable=labelField

    model_metrics = [
        model_type,
        timestamp,
        params_json,
        precision,
        recall,
        f1,
        accuracy,
        log_loss_score,
#        mae,
#        mse,
        roc_auc,
        test_item_count,
        run_time,
        rf_feature_importances,
        file_name,
        row_count,
        col_count,
        target_variable,
        analysis_lines
    ]


    # In[447]:


    model_metrics_columns = [
        'model_type',
        'timestamp',
        'params_json',
        'precision',
        'recall',
        'f1',
        'accuracy',
        'log_loss_score',
#        'mae',
#        'mse',
        'roc_auc',
        'test_item_count',
        'run_time',
        'rf_feature_importances',
        'file_name',
        'row_count',
        'col_count',
        'target_variable',
        'analysis_lines'
    ]

    metrics_df = dict(zip(model_metrics_columns, model_metrics))

    metrics_df_path = dash_data_path + 'metrics_df'
    pickle_save(metrics_df_path, metrics_df)
    perm_feature_wt_path = dash_data_path + 'perm_feature_wt'
    pickle_save(perm_feature_wt_path, perm_feature_wt)
    rand_prediction_path = dash_data_path + 'rand_prediction'
    pickle_save(rand_prediction_path, rand_prediction_pre_dummy)
    perm_feature_wt_predummy_path = dash_data_path + 'perm_feature_wt_predummy'
    pickle_save(perm_feature_wt_predummy_path, perm_feature_wt_predummy)
    featureMetricsPreDummy_path = dash_data_path + 'featureMetricsPreDummy'
    pickle_save(featureMetricsPreDummy_path, featureMetricsPreDummy)
    fpr_path = dash_data_path + 'fpr'
    pickle_save(fpr_path, list(fpr))
    tpr_path = dash_data_path + 'tpr'
    pickle_save(tpr_path, list(tpr))
#    rf_explanation_example_path = dash_data_path + 'rf_explanation_example'
#    pickle_save(rf_explanation_example_path, rf_explanation_example)
    featureMetrics_path = dash_data_path + 'featureMetrics'
    pickle_save(featureMetrics_path, featureMetrics)
    conf_matrix_path = dash_data_path + 'conf_matrix'
    pickle_save(conf_matrix_path, conf)
