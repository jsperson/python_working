#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:46:03 2019

@author: ryanbasques
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from app import app
import plotly.graph_objs as go

from Pages.create_ini import create_ini
import Pages.RunAll
from Pages.get_client_data_filenames import get_client_data_filenames
from Pages.data_health_score import data_health_score
import pickle
import json
import pandas as pd

def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK, "rb") as f:
        temp_item = pickle.load(f)
    return temp_item

def pickle_save(name, item):
    PIK = str(name)+ ".pickle"
    with open(PIK,"wb") as f:
        pickle.dump(item, f)

layout = html.Div([
        html.Div("Update", id="update-button", style={'display': 'none'}), #hidden div will update when page directed to
        html.Div(id='best-model-load-ingestion', style={'display': 'none'}), #intermediate best model metrics load - hidden div
        html.Div(id='intermediate-data-load', style={'display': 'none'}),
        html.Div(id='client-data-filenames-load', style={'display': 'none'}), #load names of all current files in client data directory
        html.Div(id='data-field-names-load', style={'display': 'none'}), #used for blacklist
        html.Div(id='dummy-div', style={'display':'none'}), #dummy hidden div for blacklst save-out output
        html.Div(id='target-variable-values-load', style={'display': 'none'}), #store target variable values

        dcc.Markdown(
                '''
##### Ingestion settings
                '''
                ),

        dcc.Link('Go to dashboard', href='/dashboard'),
        html.Br(),
        html.Br(),

        html.Div([
            html.Div([
                html.Div('File type:'),
                dcc.Dropdown(id='file-type-dropdown', options=[
                    {'label': 'Comma Separated Values (.csv)', 'value': 'csv'},
                    {'label': 'Excel (.xlsx)', 'value': 'excel'},
                    {'label': 'SQL Database', 'value': 'sql'}
                ], value='csv'),
            ], className='three columns'),
            html.Div(id='flat-file-controls-container', children=[
                html.Div('File name:'),
                dcc.Dropdown(
                    placeholder='Select file name...',
                    id='file-name'
                    )
            ], className='five columns'),
        ], className='row'),

        html.Div(id='data-health-score'),

        html.Div(id='sql-controls-container', children=[
                html.Div('Protocol:'),
                dcc.Input(
                    placeholder='postgresql',
                    type='text',
                    value='',
                    id='protocol'
                    ),
                html.Div('DB Username:'),
                dcc.Input(
                    placeholder='db_username',
                    type='text',
                    value='',
                    id='db-username'
                    ),
                html.Div('DB Password:'),
                dcc.Input(
                    placeholder='db_password',
                    type='text',
                    value='',
                    id='db-password'
                    ),
                html.Div('DB Host:'),
                dcc.Input(
                    placeholder='db_host',
                    type='text',
                    value='',
                    id='db-host'
                    ),
                html.Div('DB Port:'),
                dcc.Input(
                    placeholder='5432',
                    type='text',
                    value='',
                    id='db-port'
                    ),
                html.Div('DB Name:'),
                dcc.Input(
                    placeholder='db_name',
                    type='text',
                    value='',
                    id='db-name'
                    ),
                html.Div('Table Name:'),
                dcc.Input(
                    placeholder='table_name',
                    type='text',
                    value='',
                    id='table-name'
                    )
        ]),

        dcc.Markdown(
                '''
&nbsp;
&nbsp;
##### Runtime settings
                '''
                ),

        html.Div('NA Handling:'),

        dcc.RadioItems(options=[
            {'label': 'DROP (remove rows with null values)', 'value': 'drop'},
            {'label': 'MEAN INTERPOLATE (null values will be replaced with average from other rows)', 'value': 'mean'}
        ],
            value='drop',
            id='na_handling'),
        html.Br(),

        html.Div('Optimization Metric:'),

        dcc.RadioItems(options=[
            {'label': 'ACCURACY (overall model accuracy)', 'value': 'accuracy'},
            {'label': 'PRECISION (minimize false positives)', 'value': 'precision'},
            {'label': 'RECALL (minimize false negatives)', 'value': 'recall'},
            {'label': 'F1 (balance precision and recall)', 'value': 'f1'}
        ],
            value='accuracy',
            id='scoring_metric'),
        html.Br(),

        html.Div('Oversampling:'),

        dcc.RadioItems(options=[
            {'label': 'FALSE (do not perform oversampling - use dataset as is)', 'value': 'false'},
            {'label': 'TRUE (perform oversampling to produce balanced dataset)', 'value': 'true'}
        ],
            value='false',
            id='oversample'),
        html.Br(),

#        html.Div('Training:'),
        html.Div(dcc.RadioItems(options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
                ],
            value='True',
            id='training'), style={'display':'none'}), #hiding this for now since we don't use Training anywhere
#        html.Br(),

        html.Div([
            html.Div([
                    html.Div('Field of interest/classification:'),
#                    dcc.Input(
#                        placeholder='Enter field name...',
#                        type='text',
#                        value='',
#                        id='label_field'
#                    )
                    dcc.Dropdown(
                        placeholder='Select field name...',
                        id='label_field'
                    )
                    ], className="three columns"
            ),
            html.Div([
                html.Div('Positive label as found in field of interest:'),
                dcc.Dropdown(
                    placeholder='Select positive label...',
                    id='pos_label'
                )], className="five columns"
            ),
#            html.Div([
#                html.Div('Negative label as found in field of interest:'),
#                dcc.Input(
#                    placeholder='False',
#                    type='text',
#                    value='',
#                    id='neg_label'
#                )], className="four columns"
#            ),
        ], className="row"),

        html.Br(),

        html.Div('Ignore Time Series:'),
        dcc.RadioItems(options=[
                {'label': 'FALSE (perform time series analysis)', 'value': 'false'},
                {'label': 'TRUE (time series variables will be ignored and "Time Series Analysis" tab will be empty)', 'value': 'true'}
            ],
            value='false',
            id='ts_ignore'),
        html.Br(),

        html.Div('Ignore Market Baskets:'),
        dcc.RadioItems(options=[
                {'label': 'FALSE (perform market basket analysis)', 'value': 'false'},
                {'label': 'TRUE (market basket analysis will be ignored)', 'value': 'true'}
            ],
            value='false',
            id='mb_ignore'),
        html.Br(),

    #    html.Div(id='time-series-options-container', children=[
    #        html.Details([
    #            html.Summary('Time Series Options'),
    #            html.Div('Time Series Options Go Here')
    #        ])
    #    ]),
    #    html.Br(),

        html.Div('Blacklisted columns:'),
        html.Div('Select columns to exclude from analysis',
                 style={'fontStyle': 'italic'}),
        html.Div(
            dcc.Checklist(
                values=[],
                id='blacklist-checklist'
            ), style={'overflowY': 'scroll', 'height': 150}
        ),
        html.Br(),

        html.Button('Submit Data', id='submit-data-button', n_clicks=0),
        html.Br(),
        html.Div(id='data-submit-confirmation-output', style={'display':'none'}),
        dcc.ConfirmDialog(
                id='data-submit-confirmation',
                message="Successfully loaded data options.",
                displayed=False
            ),
        html.Br(),
        html.Button('Run Dashboard (may take a few minutes)', id='run-dashboard-button', n_clicks=0),
        html.Div(id='run-dashboard-confirmation-output', style={'display':'none'}),
        dcc.ConfirmDialog(
            id='run-dashboard-confirmation',
            message="All modules finished running successfully.",
            displayed=False
        ),
        html.Br(),
        html.Div([
            html.Div(id='model-accuracy-info', style={'fontSize': '15', 'fontWeight': 'bold'})
                ]),
        dcc.Link('Go to dashboard', href='/dashboard'),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
])

@app.callback(Output('flat-file-controls-container', 'style'), [Input('file-type-dropdown', 'value')])
def toggle_flat_file_options(file_type):
    if file_type == 'sql':
        return {'display': 'none'}
    else:
        return {'display': 'block'}

@app.callback(Output('sql-controls-container', 'style'), [Input('file-type-dropdown', 'value')])
def toggle_sql_options(file_type):
    if file_type == 'csv' or file_type == 'excel':
        return {'display': 'none'}
    else:
        return {'display': 'block'}

# toggle time series options
@app.callback(Output('time-series-options-container', 'style'),
              [Input('ts_ignore', 'value')])
def toggle_time_series_options(ts_ignore):
    if ts_ignore == 'true':
        return {'display': 'none'}
    else:
        return {'display': 'block'}

#collect all values from screen for new .ini creation
@app.callback(Output('intermediate-data-load', 'children'),
              [Input('file-type-dropdown', 'value'),
               Input('file-name', 'value'),
               Input('protocol', 'value'),
               Input('db-username', 'value'),
               Input('db-password', 'value'),
               Input('db-host', 'value'),
               Input('db-port', 'value'),
               Input('db-name', 'value'),
               Input('table-name', 'value'),
               Input('na_handling', 'value'),
               Input('scoring_metric','value'),
               Input('oversample','value'),
               Input('training', 'value'),
               Input('label_field', 'value'),
               Input('pos_label', 'value'),
#               Input('neg_label', 'value'), #superseded by data validation steps in Module 2
               Input('ts_ignore', 'value'),
               Input('mb_ignore', 'value')])
def update_data_options(file_type, file_name, protocol, db_username, db_password,
                        db_host, db_port, db_name, table_name, na_handling,
                        scoring_metric, oversample, training, label_field, pos_label,
#                        neg_label,
                        ts_ignore, mb_ignore):
    if file_name == None:
        file_name = ''
    options_dict = {'datatype': file_type,
                    'file_name': './client data/' + file_name,
					'dash_data_path': './dashboard data/',
                    'protocol':protocol,
                    'db_uname':db_username,
                    'db_pass':db_password,
                    'db_host':db_host,
                    'db_port':db_port,
                    'db_name':db_name,
                    'table_name':table_name,
                    'na_handling':na_handling,
                    'scoring_metric':scoring_metric,
                    'oversample':oversample,
                    'training':training,
                    'label_field':label_field,
                    'pos_label':pos_label,
                    'neg_label':'', #superseded by data validation steps in Module 2
                    'ts_ignore':ts_ignore,
                    'mb_ignore':mb_ignore}
    return str(options_dict)

#callback for submit data button
@app.callback(Output('data-submit-confirmation', 'displayed'),
              [Input('submit-data-button', 'n_clicks')],
              [State('intermediate-data-load', 'children')])
def submit_data_button_clicked(click, options_dict):
    if click > 0:
        if create_ini(options_dict) == True:
            return True
        else:
            return False

@app.callback(Output('data-submit-confirmation-output', 'children'),
              [Input('data-submit-confirmation', 'submit_n_clicks')])
def update_submit_data_output(submit_n_clicks):
    if submit_n_clicks:
        return 'Data successfully submitted.'

#callback for run dashboard button
@app.callback(Output('run-dashboard-confirmation', 'displayed'),
              [Input('run-dashboard-button', 'n_clicks')])
def run_dashboard_button_clicked(click):
    #run RunAll.py (runs modules 1-4)
    if click > 0:
        if Pages.RunAll.RunAllModules() == True:
            print("Modules successfully finished running.")
            return True
        else:
            print("There was an error running the modules.")
            return False

@app.callback(Output('run-dashboard-confirmation-output', 'children'),
              [Input('run-dashboard-confirmation', 'submit_n_clicks')])
def update_modules_finished_output(submit_n_clicks):
    if submit_n_clicks:
        return 'Modules successfully finished running.'

#load best model data
@app.callback(Output('best-model-load-ingestion', 'children'),
              [Input('run-dashboard-confirmation', 'displayed')])
def load_best_model_data(confirmation):
    if confirmation == False:
        best_model_data = pickle_load('./dashboard data/metrics_df') #this is a dictionary
        return json.dumps(best_model_data, default=str)

@app.callback(Output('model-accuracy-info', 'children'),
              [Input('best-model-load-ingestion', 'children')])
def show_model_accuracy(model_metrics):
    if model_metrics is not None:
        model_metrics = json.loads(model_metrics)
        accuracy = "{0:.0%}".format(model_metrics['accuracy'])
        if model_metrics['accuracy'] == 1:
            return "Model accuracy: {accuracy}. WARNING - data leakage.".format(accuracy=accuracy)
        else:
            return "Model accuracy: {accuracy}".format(accuracy=accuracy)

#collect names of files currently in client data folder
@app.callback(Output('client-data-filenames-load', 'children'),
              [Input('update-button', 'children')])
def load_client_filenames(update):
    filenames_list = get_client_data_filenames()
    return json.dumps(filenames_list)

#display client data files in dropdown
@app.callback(Output('file-name', 'options'),
              [Input('client-data-filenames-load', 'children')])
def display_client_filenames(filenames):
    filenames = json.loads(filenames)
    filenames_as_options = [{'label':x, 'value':x} for x in filenames]
    return filenames_as_options

#collect fields in currently selected data file
@app.callback(Output('data-field-names-load', 'children'),
              [Input('file-name', 'value')])
def load_data_field_names(file_name):
    file_fields = []
    if file_name != None:
        if file_name.endswith('.csv'): #only supports CSV for the moment
            file_fields = list(pd.read_csv('./client data/'+file_name,nrows=1).columns)
    return json.dumps(file_fields)

#collect target variable unique values in currently selected data file
@app.callback(Output('target-variable-values-load', 'children'),
              [Input('file-name', 'value'),
               Input('label_field', 'value')])
def load_target_field_values(file_name, label_field):
    field_values = []
    if file_name != None and label_field != None and label_field != '':
        if file_name.endswith('.csv'): #only supports CSV for the moment
            field_values = list(pd.read_csv('./client data/'+file_name)[label_field].unique())
            field_values = [str(x) for x in field_values]
    return json.dumps(field_values)

#display chosen data file fields in blacklist checklist
@app.callback(Output('blacklist-checklist', 'options'),
              [Input('data-field-names-load', 'children')])
def display_data_field_names(field_names):
    if field_names != None:
        field_names = json.loads(field_names)
        field_names_as_options = [{'label':x, 'value':x} for x in field_names]
        return field_names_as_options
    else:
        return []

#save out blacklist fields to be read by module 2
@app.callback(Output('dummy-div', 'children'),
              [Input('submit-data-button', 'n_clicks')],
              [State('blacklist-checklist', 'values')])
def save_blacklist_fields(n_clicks, blacklist_fields):
    if n_clicks > 0:
        pickle_save('./dashboard data/blacklist_fields', blacklist_fields)
    return None

#display chosen data file fields in positive class dropdown
@app.callback(Output('pos_label', 'options'),
              [Input('target-variable-values-load', 'children')])
def display_target_field_values(target_values):
    if target_values != None:
        target_values = json.loads(target_values)
        target_values_as_options = [{'label':x, 'value':x} for x in target_values]
        return target_values_as_options
    else:
        return []

#display chosen data file fields in target field dropdown
@app.callback(Output('label_field', 'options'),
              [Input('data-field-names-load', 'children')])
def display_possible_target_field_names(field_names):
    if field_names != None:
        field_names = json.loads(field_names)
        field_names_as_options = [{'label':x, 'value':x} for x in field_names]
        return field_names_as_options
    else:
        return []

#send data to data health score calculator and display score
@app.callback(Output('data-health-score', 'children'),
              [Input('file-name', 'value')])
def calc_data_health_score(file_name):
    health_score_info = None
    if file_name != None:
        if file_name.endswith('.csv'): #only supports CSV for the moment
            df = pd.read_csv('./client data/'+file_name)
            health_score_info = data_health_score(df)
    if health_score_info:
        #put together string
#        recs = [html.Div(x) for x in health_score_info[1]]
        recsFromDict = []
        #collapsible divs with each category of recommendations
        if len(health_score_info[2]) > 0:
            for key in health_score_info[2]:
                keyTitle = key
                if key == 'rowNumScore':
                    keyTitle = 'Row count'
                elif key == 'dataTypeMixScore':
                    keyTitle = 'Data type mix'
                elif key == 'categoricalBlowoutScore':
                    keyTitle = 'Categorical blowout'
                elif key == 'data_sparsity_score':
                    keyTitle = 'Data sparsity'

                detailDivs = [html.Div(x) for x in health_score_info[2][key]]
                detailDiv = html.Div(detailDivs)

                recsFromDict.append(
                    html.Details([
                        html.Summary(keyTitle),
#                        html.Div(health_score_info[2][key])
                        detailDiv
                    ])
                        )

        #create pie chart
        data = [
            {
                'values': [100-health_score_info[0],health_score_info[0]],
                'labels': ['-', '+'],
                'textinfo' : 'label+percent',
                'type': 'pie',
                'marker':{
                        'colors':['#ffbfbf', '#8cf783'],
                        'line':{'color':'#000000', 'width':1}
                                  },
            },
        ]
        health_score = html.Div([
        dcc.Markdown(
                '''
&nbsp;
##### Data Health
                '''
                ),
        html.Div([
                    html.Div(dcc.Graph(
                        id='graph',
                        figure={
                            'data': data,
                            'layout': {
                                'margin': {'l': 30,'r': 0,'b': 0,'t': 22},
                                'showlegend': False
                            }
                        },
                        style={'height': 140, 'width': 140},
                    ), className="one column"),
#                html.Div(recs, style={'overflowY': 'scroll', 'height': 130, 'fontStyle': 'italic', 'width':1000,
#                        'marginTop':10,'marginLeft':100}, className="three columns"),
                html.Div('Recommendations:', style={'fontStyle': 'italic',
                                                      'marginTop':22,'marginLeft':100, 'fontSize': 18}, className="three columns"),
                html.Div(recsFromDict, style={'fontStyle': 'italic', 'width':1000,
                        'marginTop':5,'marginLeft':110}, className="three columns"),
                    ], className="row")
        ])
        return health_score
