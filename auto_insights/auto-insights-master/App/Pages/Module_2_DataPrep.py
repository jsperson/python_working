
# coding: utf-8

# In[1]:


from Pages.Module_1_DataIngestion import generateDf
import configparser
import pandas as pd
import pickle
from datetime import datetime
from datetime import timedelta
from patsy import dmatrices
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re

def pickle_save(name, item):
    PIK = str(name)+ ".pickle"
    with open(PIK,"wb") as f:
        pickle.dump(item, f)

def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK, "rb") as f:
        temp_item = pickle.load(f)
    return temp_item

def stripNonAlphanumeric(items): #(keeps underscores)
    items = [x.strip() for x in items] #strip whitespace on left or right
    items = [re.sub(' |-', '_', x) for x in items] #replace spaces and dashed with unnderscore
    items = [re.sub('\W|^_', '', x) for x in items] #keep only number, letters, and underscores
    items = ["_"+x if x[0].isdigit() else x for x in items] #if first character is number, prefix with "_"
    return items

def returnCleanDf():

    config = configparser.ConfigParser()
    config.read('./ml_box.ini')
    runtime_settings = config['RUNTIME']
    labelField = runtime_settings['label_field']
    dash_data_path = runtime_settings['dash_data_path']

    #clean labelField
    labelField = stripNonAlphanumeric([labelField])[0]

    def initData():

        raw = generateDf()
        return raw

    def prepData():
        raw = initData()

        #data stats
        print('\nData loaded.')
        print(f'\nSize: {raw.shape[0]} rows, {raw.shape[1]} columns/fields.')

        df = raw.copy()
        return df

    #utility to transform target variable to numeric
    def transformLabel(df, labelField, printOut):
        pos_label = runtime_settings['pos_label']
#        neg_label = runtime_settings['neg_label']

        #set neg_label to the only value that isn't pos_label
        all_labels = list(df[labelField].unique())
        all_labels = [str(x) for x in all_labels]
        all_labels.remove(pos_label)
        neg_label = all_labels[0]

        #convert to int
        try:
            pos_label = int(pos_label)
            neg_label = int(neg_label)
        except:
            None


        #Make sure that pos_label and neg_label are found in the labelField
#        assert pos_label in df[labelField].unique(), pos_label+" not found in "+labelField
#        assert neg_label in df[labelField].unique(), neg_label+" not found in "+labelField

        #Make sure that labelField has exactly 2 unique values
        uniqueValues = df[labelField].nunique()
        assert uniqueValues<3, labelField+" has more than two unique values."
        assert uniqueValues>1, labelField+" only has one unique value."

        if printOut:
            print('\nRecoding to positive labels: %s' % (pos_label))
            print('\nRecoding to negative labels: %s' % (neg_label))

        df[labelField].replace(neg_label,0,inplace=True)
        df[labelField].replace(pos_label,1,inplace=True)
        return df

    #remove fields that user blacklisted on ingestion page
    def removeBlacklistedFields(df):
        blacklist_path = dash_data_path + 'blacklist_fields'
        blacklist = pickle_load(blacklist_path)
        df.drop(blacklist, axis=1, inplace=True)
        return df

    # test if a column can be interpreted as a date
    def is_datetime(s):
        try:
            # convert to datetime
            pd.to_datetime(s)

            # if column is 'object' datatype, return True
            if s.dtypes == 'object':
                return True

            # otherwise, test numeric data
            else:
                # range of dates
                min_date = pd.to_datetime(min(s))
                max_date = pd.to_datetime(max(s))
                date_diff = max_date - min_date

                # if the range of dates is > 1 day, return True
                if (date_diff > timedelta(days=1)) & (min_date > pd.to_datetime(0)):
                    return True
                else:
                    return False
        except:
            return False

    # converts float number representing number of hours into h, m, s
    def floatHourToTime(fh):
        h, r = divmod(fh, 1)
        m, r = divmod(r, 1/60)
        return (
            int(h),
            int(m),
            int(r*60*60),
        )

    # converts Excel dates stored as number to datetime
    def excelNumToDateTime(s):
        if isinstance(s, float):
            try:
                dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(s) - 2)
                hour, minute, second = floatHourToTime((s % 1)*24)
                dt = dt.replace(hour=hour, minute=minute, second=second)
                return dt
            except:
                return np.nan
        else:
            try:
                dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + s - 2)
                return dt
            except:
                return np.nan

    # test if string can be interpreted as a dayfirst date
    def is_dayfirst(d):
        # if object is not a string, return False
        if isinstance(d, str) == False:
          return False

        else:
            # split string at ' ' and take the first component (the date component)
            d_str = d.split()[0]

            # convert potential separators to '/'
            new_d = d_str.replace('-','/').replace('.','/').replace(',','/')
            try:
                # test if this string can be interpreted as DDMMYYYY
                datetime.strptime(new_d, "%d/%m/%Y")
                return True
            except ValueError:
                try:
                    # test if this string can be interpreted as DDMMYY
                    datetime.strptime(new_d, "%d/%m/%y")
                    return True
                except ValueError:
                    return False

    # test if string can be interpreted as a monthfirst date
    def is_monthfirst(d):
        # if object is not a string, return False
        if isinstance(d, str) == False:
          return False

        else:
            # split string at ' ' and take the first component (the date component)
            d_str = d.split()[0]

            # convert potential separators to '/'
            new_d = d_str.replace('-','/').replace('.','/').replace(',','/')
            try:
                # test if this string can be interpreted as DDMMYYYY
                datetime.strptime(new_d, "%m/%d/%Y")
                return True
            except ValueError:
                try:
                    # test if this string can be interpreted as DDMMYY
                    datetime.strptime(new_d, "%m/%d/%y")
                    return True
                except ValueError:
                    return False

    # determine if column should be parsed as dayfirst or monthfirst
    def parse_as_dayfirst(df_column):
        dayfirst_count = 0
        monfirst_count = 0

        # count number of items in the column that can be interpreted as dayfirst vs. monthfirst
        for index, item in df_column.iteritems():
            if is_dayfirst(item) == True:
                dayfirst_count += 1
            if is_monthfirst(item) == True:
                monfirst_count += 1

        # if more strings can be interpreted as dayfirst, return True
        if dayfirst_count > monfirst_count:
            return True
        # otherwise return false
        else:
            return False

    def cols_for_ts_analysis(df, date_cols, non_date_cols):
        # date columns
        date_cols_for_ts = date_cols.copy()
        for col in date_cols:
            # max gap in dates
            max_gap = df[col].sort_values().diff().max()

            # percent null
            perc_null = (df[col].isna().sum())/df[col].size

            # if max gap > 365 days or more than 50% null, do not use for time series analysis
            if (max_gap > timedelta(days=365)) | (perc_null > .5):
                date_cols_for_ts.remove(col)

        # non-date columns
        non_date_cols_for_ts = non_date_cols.copy()
        for col in non_date_cols:
            # remove very-low-support (<5%) and very-high-support (>95%) indicators from time-series iterations
            if ((df[col]==1).sum() + (df[col]==0).sum() == df[col].shape[0]):
                if (df[col].mean()<.05) | (df[col].mean()>.95):
                    non_date_cols_for_ts.remove(col)

        return (date_cols_for_ts, non_date_cols_for_ts)


    '''
    Will take a high cardinality column, determine if a numeric column with non-numeric cells, and correct for common
    errors ('na', '123', ' ').
    '''
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def correctNumericCol(df_column, c):
        num_count = 0
        non_num_count = 0
        non_nums = []

        for index, item in df_column.iteritems():

            if is_number(item) == True:
                num_count += 1
            else:
                non_num_count += 1
                non_nums.append(item)

        non_nums = set(non_nums) #all unique non-numeric entries in column

        # need some way to see how many are not numeric, set some assumption cut-off
        # (more than X% can't be converted to a number, therefore we don't believe this is a numeric column)
        if non_num_count/(num_count+non_num_count) <=0.1:
    #        print(non_num_count)
    #        print(num_count)
            if non_num_count > 0:
                for non_num in non_nums:
                    if non_num in [' ', '']:
                        new_df_column = df_column.replace(non_num, float('nan'))
                        new_df_column = pd.to_numeric(new_df_column)
                        return new_df_column
                    else:
                        print('\nHave not implemented correction for:', non_num)
            else:
                return pd.to_numeric(df_column) #if already stripped "$" and this was the only issue
        else:
            print('\n',c, 'is most likely not a numeric column...')
            print('\nDROPPING', c, '\n')
            return False #alert below to drop

    # test if numeric column is an Excel-style date number
    def isExcelDateNumberColumn(df_column, c):
        try:
            if correctNumericCol(df_column, c) == False:
                return False
            else:
                df_column = correctNumericCol(df_column, c)
                print('\nNumeric data detected in', c)
                print('\nTesting if', c, 'is an Excel date number...')
        except ValueError:
            df_column = correctNumericCol(df_column, c)
            print('\nNumeric data detected in', c)
            print('\nTesting if', c, 'is an Excel date number...')

            dt_count = 0
            non_dt_count = 0
            num_count = 0
            non_num_count = 0
            for index, item in df_column.iteritems():
                if pd.isnull(item):
                    non_num_count += 1
                else:
                    num_count += 1
                    if pd.isnull(excelNumToDateTime(item)):
                        non_dt_count += 1
                    else:
                        dt_count += 1

            # test if at least 90% of the numbers can convert to dates
            if num_count > 0:
                if dt_count/num_count >= .9:
                    min_date = excelNumToDateTime(np.nanmin(df_column))
                    max_date = excelNumToDateTime(np.nanmax(df_column))
                    try:
                        date_diff = max_date - min_date
                    except:
                        print('\nColumn', c, 'does not appear to be an Excel date number. Leaving as numeric.')
                        return False
    
                    # if the range of dates is > 1 day, return True
                    if (date_diff > timedelta(days=1)):
                        if (min_date > pd.to_datetime(0)) & (max_date < (datetime.now() + timedelta(days=3650))):
                            print('\nColumn', c, 'appears to be an Excel date number.')
                            return True
                        else:
                            print('\nColumn', c, 'does not appear to be an Excel date number. Leaving as numeric.')
                            return False
                    else:
                        print('\nColumn', c, 'does not appear to be an Excel date number. Leaving as numeric.')
                        return False
                    return False
                else:
                    print('\nColumn', c, 'does not appear to be an Excel date number. Leaving as numeric.')
                    return False
            else:
                print('\nColumn', c, 'does not appear to be an Excel date number. Leaving as numeric.')
                return False

    # convert Excel Date number column to datetime
    def convertExcelDateNumberColumn(df_column):
        new_df_column = df_column - 7
        for index, item in df_column.iteritems():
            new_df_column[index] = excelNumToDateTime(item)
        return new_df_column

    def dropSparseColumns(df, threshold=0.15): #threshold is null % we start dropping columns at
        no_datetime_df = df.select_dtypes(exclude=[np.datetime64]) #ignore datetime variables

        sparsityPerColumn = no_datetime_df.isna().sum()/no_datetime_df.shape[0]
        colsToDrop = list(sparsityPerColumn[sparsityPerColumn>threshold].index)

        print("\nIdentifying sparse columns to drop. Threshold =", '{0:.0%}'.format(threshold))

        for index, value in sparsityPerColumn.iteritems():
            if value > threshold:
                print("\nDROPPING", index, "because it contains", '{0:.0%}'.format(value), "percent nulls.")

        df = df.drop(colsToDrop, axis=1)
        return df
    
    #drop categorical variables with >X unique values
    #(or else each unique value gets blown out into a new column post-dummification)
    def dropBlowoutColumns(df, threshold=100):
        #high-cardinality categorical fields
        categoricalCols = list(df.select_dtypes(exclude=[np.number, np.datetime64]).columns.values)
        for c in categoricalCols:
            if df[c].nunique() > threshold:
                df = df.drop(c, axis=1)
                print("\n", c, 'has more than', threshold, 'unique values. Dropping to prevent blowout.')
        
        return df

    def cleanColumnNames(df): #gets rid of non-alphanumeric chars (except for _)

        cols = stripNonAlphanumeric(df.columns)
        df.columns = cols

        return df

    def stripOutSymbols(df): #get rid of specific symbols - for now, just "$"
        for column in df:
            if df[column].dtype == 'object':
                try:
                    df[column] = df[column].str.replace('$', '')
                except Exception as error:
                    print(column, ":", error)
        return df

    def handleDateTimeCols(df):
        # get file type
        config = configparser.ConfigParser()
        config.read('./ml_box.ini')
        ingest_settings = config['INGEST']
        file_type = ingest_settings['datatype']

        for c in df.columns:
            # test for datetime type data in strings
            if df[c].dtypes == 'object':
                if is_datetime(df[c]) == True:
                    print('\nColumn', c, 'appears to be datetime data.')
                    dayfirst=parse_as_dayfirst(df[c])

                    # convert to datetime
                    if dayfirst == True:
                        print('\nInterpreting', c, 'as a day-first date string.')
                        df[c] = pd.to_datetime(df[c], dayfirst=dayfirst)
                    else:
                        df[c] = pd.to_datetime(df[c])
                    print("\nConverted", c, "to a datetime column.\n")
            # test for datetime type data in numbers
            else:
                # if input file is csv or excel, look for excel-stype datetime numbers
                if (file_type == 'csv') | (file_type == 'excel'):
                    if isExcelDateNumberColumn(df[c],c) == True:
                        df[c] = convertExcelDateNumberColumn(df[c])
                        print("\nConverted", c, "to a datetime column.\n")
                # otherwise, test for python-stype datetime numbers
                else:
                    if is_datetime(df[c]) == True:
                        print('\nColumn', c, 'appears to be datetime data.')
                        df[c] = pd.to_datetime(df[c])
                        print("\nConverted", c, "to a datetime column.\n")
        return df


    #Exclude high cardinality data (ids). If count unique of column == number of rows, exclude.
    #Note that a high % (90+) may indicate a numeric column incorrectly cast as object
    def addressHighCardinality(df):
        num_rows = df.shape[0]
        for c in df.columns:
            count_unique_pct = df[c].nunique()/num_rows

            #test
    #         print(c, 'has', "{0:.0%}".format(count_unique_pct), 'cardinality.')
            if (count_unique_pct >= 0.98) & (df[c].dtypes != 'datetime64[ns]'): #if >98% cardinality
                print('\n', c, 'has', "{0:.0%}".format(count_unique_pct), 'cardinality. Dropped from dataframe.\n')
                df = df.drop(labels=c, axis=1)

            elif (count_unique_pct >= 0.1) & (df[c].dtypes != 'datetime64[ns]'): #warn if high cardinality otherwise
                print('\nWARNING:', c, 'has', "{0:.0%}".format(count_unique_pct), 'cardinality. Would result in', df[c].nunique(), 'dummy columns. Is this a numeric column with non-numeric inputs?')
                print('Attempting to correct for non-numeric data.')
                try:
                    if correctNumericCol(df[c], c) == False:
                        df = df.drop(labels=c, axis=1)
                    else:
                        df[c] = correctNumericCol(df[c], c)
                        print("\nFixed", c, "to a numeric column.\n")
                except ValueError:
                    df[c] = correctNumericCol(df[c], c)
                    print("\nFixed", c, "to a numeric column.\n")

    #        elif count_unique_pct >= 0.1: #warn if high cardinality otherwise
    #            print('WARNING:', c, 'has', "{0:.0%}".format(count_unique_pct), 'cardinality. Would result in', df[c].nunique(), 'dummy columns.')
    #            print('DROPPING', c, '\n')
    #            df = df.drop(labels=c, axis=1)

        return df

    def findMulticollinearVars(df, threshold=10):

        print("\nChecking for multicollinearity using Variance Inflation Factor (VIF).\n")

        #Data prep
        temp_df = df.copy()
        temp_df = transformLabel(temp_df, labelField, False)
#        temp_df.dropna(inplace=True)
        temp_df = temp_df._get_numeric_data() #drop non-numeric cols

        #gather features
        x_cols = list(temp_df.columns)

        #take out dependent var
        x_cols.remove(labelField)
        
        if len(x_cols) > 1:
            features = "+".join(x_cols)
    
            # get y and X dataframes based on this regression:
            y, X = dmatrices(labelField + '~' + features, temp_df, return_type='dataframe')
    
            varsToDelete = [] #will hold found multicollinear vars
            largeVIF = True #initiate loop trigger
            variables = list(range(X.shape[1])) #index of variables
            columns = list(X.columns) #names of variables
    
            #while loop until largest vif <= threshold
            while largeVIF == True:
    
                #get VIF for all vars
                vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
                largest_vif = max(vif)
    
                if columns[vif.index(largest_vif)] != "Intercept": #user doesn't know/care about Intercept
                    print("Largest VIF present:", round(largest_vif, 0), "Variable:", columns[vif.index(largest_vif)])
    
                #if max > threshold, continue (multicollinearity is present in one or more vars)
                if largest_vif > threshold:
    
                    #if largest_vif is not infinity, drop. else go to one_hot method to weed out problem variable
                    if not np.isinf(largest_vif):
                        maxloc = vif.index(largest_vif)
                        if columns[maxloc] != "Intercept":
                            print("Deleting variable:", columns[maxloc], '\n')
                        del variables[maxloc]
                        varsToDelete.append(columns[maxloc])
                        del columns[maxloc]
    
                    else:
                        #find one-hot mean VIFs for each variable
                            #for each variable, remove it from the data and calculate VIFs on all remaining variables.
                            #take the average of the VIFs on all remiaining variables
                            #the lowest average means that the associated variable that was taken out is the problem causer
                        one_hot_scores = []
                        for v in variables:
                            #take v out of X
                            one_hot_variables = variables.copy()
                            one_hot_variables.remove(v)
    
                            #get VIFs with this one hot list
                            one_hot_vifs = [variance_inflation_factor(X.iloc[:, one_hot_variables].values, ix)
                                           for ix in range(X.iloc[:, one_hot_variables].shape[1])]
                            one_hot_vif = np.mean(one_hot_vifs)
                            one_hot_scores.append(one_hot_vif)
    
                        #drop variable with lowest one-hot mean VIF
                        minloc = one_hot_scores.index(min(one_hot_scores))
                        if columns[minloc] != "Intercept":
                            print("Deleting variable:", columns[minloc], '\n')
                        del variables[minloc]
                        varsToDelete.append(columns[minloc])
                        del columns[minloc]
    
                else: #no multicollinearity
                    largeVIF = False
                    print("\nNo multicollinearity over", threshold, "VIF present.")
    
            if 'Intercept' in varsToDelete:
                varsToDelete.remove('Intercept')
    
            #Get rid of [T.True] suffix (is added due to patsy reading booleans as categorical)
            varsToDelete = [re.sub('\[T.True\]|\[T.False\]', '', x) for x in varsToDelete]
    
            return varsToDelete
        
        else:
            print("Not enough categorical columns to perform VIF analysis. (need >1)")
            return []

    def dropNa(df):
        config = configparser.ConfigParser()
        config.read('./ml_box.ini')
        runtime_settings = config['RUNTIME']
        if runtime_settings['na_handling'] == 'drop':
            print('\nna handling style: drop')

            #look at nan rows as % of total. if >1%, mean impute instead
            nans = df.isnull().any(axis=1).sum() #count num of rows with nan
            pct_nan = nans/df.shape[0]
            if pct_nan >= 0.01:
                print('\n>1% of dataframe rows have na - trying mean imputation. non-numeric columns will keep nas\n')

                #MEAN IMPUTE
                df.fillna(df.mean(), inplace=True)

            else:
                df.dropna(axis=0, inplace=True)
                print('\ndropped', nans, 'rows with nan values, or', pct_nan, 'of dataframe')
        elif runtime_settings['na_handling'] == 'mean':
            print('\nna handling style: mean. note that non-numeric columns will keep nas')
            #MEAN IMPUTE
            df.fillna(df.mean(), inplace=True)

        return df


    # ### Dummify categorical variables

    # In[7]:


    def categorizeFields(df, labelField):
        #Identify which data fields are categorical
        cat_fields = list(df.select_dtypes(include='object').columns)

        if labelField in cat_fields:
            cat_fields.remove(labelField) #don't dummify target label

        #Identify numeric data fields that should be categorical
        for column in list(df.select_dtypes(exclude='object').columns):
            if (df[column].nunique() <= 5) & (column != labelField): #Assumption - converts seniors (0, 1) to categorical
                print("\nField '", column, "' only has", df[column].nunique(), "unique values. Converting to categorical.\n")
                cat_fields.append(column)

        #if non-numeric field has two values, binarize, else categorize
        binary_fields = []
        for c in cat_fields:
            print('\n', c, df[c].nunique())
            if df[c].nunique() == 2:
                binary_fields.append(c)
        
        cat_fields = [x for x in cat_fields if x not in binary_fields]

        sep = ' = '
        df = pd.get_dummies(df, columns=cat_fields, prefix_sep= sep)
        df = pd.get_dummies(df, columns=binary_fields,prefix_sep= sep) #, drop_first=True)

        #create dummy variable memory - {parent_dummy: [parent_dummy = 0, parent_dummy = 1, parent_dummy = 2]}
        dummy_fields = binary_fields + cat_fields
        dummy_fields = [x+sep for x in dummy_fields]

        #save dummy variable memory structure (remember which children belong to which parents)
        dummy_memory = {}
        for d in dummy_fields:
            for col in df.columns:
                if col.startswith(d):
                    if d in dummy_memory:
                        dummy_memory[d].append(col)
                    else:
                        dummy_memory[d] = [col]
        dummy_memory_path = dash_data_path + 'dummy_memory'
        pickle_save(dummy_memory_path, dummy_memory)

        if runtime_settings['training'] == 'True':
            print("\nTraining Phase")
            df = transformLabel(df, labelField, True)
        return df

#    def returnCleanDf():
    df = prepData()
    df = removeBlacklistedFields(df)
    df = cleanColumnNames(df)
    df = stripOutSymbols(df)
    df = handleDateTimeCols(df)
    df = dropSparseColumns(df, threshold=0.15)
    df = addressHighCardinality(df)
    df = dropBlowoutColumns(df)
    df = dropNa(df)

    #Multicollinear vars
    varsToDelete = findMulticollinearVars(df)
    if len(varsToDelete)>0:
        df.drop(varsToDelete, axis=1, inplace=True)
        print("DELETED multicollinear variables:", ' '.join(varsToDelete))

    #Save out pre-dummified data for single instance prediction page
    pre_dummified_clean_df_path = dash_data_path + 'pre_dummified_clean_df'
    pickle_save(pre_dummified_clean_df_path, df)

    #dummify data
    df = categorizeFields(df, labelField)

    # save list of ALL date columns
    date_cols = list(df.select_dtypes(include=['datetime64[ns]']))
    non_date_cols = list(df.select_dtypes(exclude=['datetime64[ns]']))
    all_date_cols_path = dash_data_path + 'all_date_cols'
    pickle_save(all_date_cols_path, date_cols)
    all_non_date_cols_path = dash_data_path + 'all_non_date_cols'
    pickle_save(all_non_date_cols_path, non_date_cols)
    # save df, with date columns
    clean_df_time_path = dash_data_path + 'clean_df_time'
    pickle_save(clean_df_time_path, df)

    # save list of date columns used in TS analysis
    (ts_date_cols, ts_non_date_cols) = cols_for_ts_analysis(df, date_cols, non_date_cols)
    date_cols_path = dash_data_path + 'date_cols'
    pickle_save(date_cols_path, ts_date_cols)
    non_date_cols_path = dash_data_path + 'non_date_cols'
    pickle_save(non_date_cols_path, ts_non_date_cols)
    # save df, excluding date columns
    df = df.select_dtypes(exclude=['datetime64[ns]'])
    clean_df_path = dash_data_path + 'clean_df'
    pickle_save(clean_df_path, df)

    return df


    # In[20]:


    # returnCleanDf()
