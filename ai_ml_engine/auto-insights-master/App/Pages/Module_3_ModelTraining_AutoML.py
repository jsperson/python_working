#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Pages.Module_2_DataPrep import returnCleanDf
from tpot import TPOTClassifier
import pickle
import configparser
from sklearn.model_selection import train_test_split
import io
import sys
from contextlib import redirect_stdout
import re

def pickle_save(name, item):
    PIK = str(name)+ ".pickle"
    with open(PIK,"wb") as f:
        pickle.dump(item, f)

#Load best model
def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK,"rb") as f:
        temp_item = pickle.load(f)
    return temp_item

def stripNonAlphanumeric(items): #(keeps underscores)
    items = [x.strip() for x in items] #strip whitespace on left or right
    items = [re.sub(' |-', '_', x) for x in items] #replace spaces and dashed with unnderscore
    items = [re.sub('\W|^_', '', x) for x in items] #keep only number, letters, and underscores
    items = ["_"+x if x[0].isdigit() else x for x in items] #if first character is number, prefix with "_"
    return items

def module3():

    config = configparser.ConfigParser()
    config.read('./ml_box.ini')
    runtime_settings = config['RUNTIME']
    labelField = runtime_settings['label_field']
    labelField = stripNonAlphanumeric([labelField])[0] #clean labelField
    scoring_metric = runtime_settings['scoring_metric']
    dash_data_path = runtime_settings['dash_data_path']

    with io.StringIO() as buf, redirect_stdout(buf):
        df = returnCleanDf()
        data_cleansing_output_path = dash_data_path + 'data_cleansing_output'
        pickle_save(data_cleansing_output_path, buf.getvalue())
    redirect_stdout(sys.stdout);
    #df = returnCleanDf()

    X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:, df.columns != labelField],
                                                        df[labelField], test_size=0.2, random_state=33)

    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=33)

    #Perform data rebalancing here on training data
    oversample = runtime_settings['oversample']

    if oversample == 'true':
        print("Old value counts:")
        print(Y_train.value_counts())

        from imblearn.over_sampling import SMOTE
        import pandas as pd

        smote = SMOTE(sampling_strategy='minority')
        X_columns = X_train.columns
        X_train, Y_train = smote.fit_sample(X_train, Y_train)

        #Convert back from numpy arrays into pandas
        X_train = pd.DataFrame(X_train, columns=X_columns)
        Y_train = pd.Series(Y_train)

        print("Oversampling minority churn data.")
        print("New value counts:")
        print(Y_train.value_counts())

    print(X_train.shape[0], "training samples")
    print(X_test.shape[0], "test samples")

    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                        scoring=scoring_metric, random_state=42,
                                        verbosity=2, config_dict='TPOT light')

    pipeline_optimizer.fit(X_train, Y_train)

    best_model = pipeline_optimizer.fitted_pipeline_

    print("Best model:")
    print(best_model)

    tpot_exported_pipeline_module_path = dash_data_path + 'tpot_exported_pipeline_module.py'
    pipeline_optimizer.export(tpot_exported_pipeline_module_path)
    print("Exported python script for best model pipeline to 'tpot_exported_pipeline_module.py'")

    best_model_automl_path = dash_data_path + 'best_model_automl'
    pickle_save(best_model_automl_path, best_model)
    print("Saved best model as 'best_model_automl'.")

    #Pickle data
    X_train_path = dash_data_path + 'X_train'
    pickle_save(X_train_path, X_train)
    X_test_path = dash_data_path + 'X_test'
    pickle_save(X_test_path, X_test)
    Y_train_path = dash_data_path + 'Y_train'
    pickle_save(Y_train_path, Y_train)
    Y_test_path = dash_data_path + 'Y_test'
    pickle_save(Y_test_path, Y_test)

    print("Exported training and testing data (X_ and Y_)")
