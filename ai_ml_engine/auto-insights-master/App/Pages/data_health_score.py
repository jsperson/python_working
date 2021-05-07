#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:35:22 2019

@author: ryanbasques
"""

import numpy as np
import operator

def data_health_score(df):
    
    #Sparsity check
    nullFields = df.isna().sum()/df.shape[0]
    nullFields.sort_values(ascending=False, inplace=True)
    #inverse of total number of null cells over total number of cells
    data_sparsity_score = 100*(1-(df.isna().sum().sum() / (df.shape[0] * df.shape[1])))
    
    #Number of rows check
    numRows = df.shape[0]
    if numRows <= 1000:
        rowNumScore = numRows/20
    elif numRows <7500:
        rowNumScore = 70
    elif numRows <15000:
        rowNumScore = 90
    else:
        rowNumScore = 100
        
    #mix of variable types - check for continuous/numeric
    numericCols = list(df.select_dtypes(include=[np.number]).columns.values)
    if len(numericCols) >= 5:
        dataTypeMixScore_numeric = 100
    else:
        dataTypeMixScore_numeric = 20*len(numericCols)
        
    categoricalCols = list(df.select_dtypes(exclude=[np.number]).columns.values)
    if len(categoricalCols) >= 5:
        dataTypeMixScore_categorical = 100
    else:
        dataTypeMixScore_categorical = 20*len(categoricalCols)
        
    dataTypeMixScore = np.average([dataTypeMixScore_numeric, dataTypeMixScore_categorical])
        
    #high-cardinality categorical fields
    manyUniqueValues={}
    for c in categoricalCols:
        
        if df[c].nunique() > 100:
            manyUniqueValues[c] = df[c].nunique()
            
    if len(manyUniqueValues.keys()) > 2:
        categoricalBlowoutScore = 0
    elif len(manyUniqueValues.keys()) == 2:
        categoricalBlowoutScore = 25
    elif len(manyUniqueValues.keys()) == 1:
        categoricalBlowoutScore = 50
    else:
        categoricalBlowoutScore = 100
        
    overallHealthScore = np.average([data_sparsity_score,
                                     rowNumScore,
                                     dataTypeMixScore,
                                    categoricalBlowoutScore])
    recs=[]
    recsDict = {}
#    recs.append("Overall data health: "+"{:.0%}".format(overallHealthScore/100)+" Recommendations:")
#    recs.append('\n')
#    recs.append('Recommendations:')
    if data_sparsity_score<100:
        if max(nullFields) > 0.02:
            recString = "Try removing null fields or filling null values in your dataset (currently"+\
                  " {:.1%}".format((100-data_sparsity_score)/100)+\
                  " null). The top most null columns are:"
            recs.append(recString)
            for i, x in nullFields.items():
                if x>0.02:
                    recs.append(' '+ i +" {:.1%}".format(x))
#                    recString += ' '+ i +" {:.1%}".format(x)
            recs.append('\n')
            recsDict['data_sparsity_score'] = recs.copy()
            recs=[]
    if rowNumScore<100:
        recString = "Try adding more rows to your data. You currently have"+\
              " {:,}".format(numRows)+ " rows."
        recs.append(recString)
        recs.append('\n')
        recsDict['rowNumScore'] = recs.copy()
        recs=[]
    if dataTypeMixScore<100:
#        recString = ''
        if len(numericCols)<5:
            tempRecStr = "Try adding more numeric fields to your data. You currently have "+\
                  str(len(numericCols))+'.'
            recs.append(tempRecStr)
#            recString += tempRecStr
        if len(categoricalCols)<5:
            tempRecStr = "Try adding more categorical fields to your data. You currently have "+\
                  str(len(categoricalCols))+'.'
            recs.append(tempRecStr)
#            recString += tempRecStr
        recs.append('\n')
        recsDict['dataTypeMixScore'] = recs.copy()
        recs=[]
    if categoricalBlowoutScore<100:
        recString = "Some categorical fields have too many unique values to provide model power. \n \
Either remove them or bucket multiple values together. Fields include:"
        recs.append(recString)
        sorted_x = sorted(manyUniqueValues.items(), key=operator.itemgetter(1), reverse=True)
        for c in sorted_x:
            tempRecStr = ' '+ c[0]+ " ("+ str(c[1]) + " unique values)"
            recs.append(tempRecStr)
#            recString += tempRecStr
        recs.append('\n')
        recsDict['categoricalBlowoutScore'] = recs.copy()
        recs=[]
        
    return (overallHealthScore, recs, recsDict)