#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:22:11 2019

@author: ryanbasques
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import warnings
import pandas as pd
import pickle
#import dash_table
import json
import numpy as np
from app import app
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import Pages.time_series_analysis

#silence SettingWithCopy warnings
pd.options.mode.chained_assignment = None

#silence FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK, "rb") as f:
        temp_item = pickle.load(f)
    return temp_item

def generate_metrics_table(formatted_metrics, col_names):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in col_names])] +

        # Body
        [html.Tr([
            html.Td(col) for col in formatted_metrics
        ])]
    )

def generate_table_from_dataframe(dataframe):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )

#Given set of observations, return probability of 0 and 1 using best model
def get_proba_from_x(observation):
    try:
        best_model_pipeline = pickle_load('./dashboard data/best_model_automl')
        probabilities = best_model_pipeline.predict_proba(observation)
        return probabilities[0]
    except ValueError as error:
        print("Cannot predict probability:", error)
        return [0, 0]

#converts pre-dummy column and value to post-dummy columns and values
def preToPostDummyValueConversion(col_predummy, value_predummy, dummy_memory, sep=' = '):
    '''
    Takes in predummy field name and value, returns dict of postdummy field names and values.
    Ex. ('Partner', 'Yes') -> {'Partner = Yes': 1, 'Partner = No': 0}
    If no dummy conversion takes place, just returns same value (tenure, 71) -> {tenure: 71}.
    '''

    if col_predummy in dummy_memory.keys(): #if dummy conversion took place for this column
        col_postdummy = col_predummy+sep+str(value_predummy)
        postdummy_options = dummy_memory[col_predummy]

        post_dummy_dict = {}

        if len(postdummy_options) == 1: #binary dummy variable (we drop first for these)
            postdummy_option = postdummy_options[0]
            if col_postdummy != postdummy_option:
                post_dummy_dict[postdummy_option] = 0
            else:
                post_dummy_dict[col_postdummy] = 1
            return post_dummy_dict

        else: #multiple option dummy variable
            post_dummy_dict[col_postdummy] = 1
            for option in postdummy_options:
                if option != col_postdummy:
                    post_dummy_dict[option] = 0
            return post_dummy_dict

    return {col_predummy:value_predummy}

layout = html.Div([
    html.H3('AutoInsight Report'),
    dcc.Link('Go to data entry page', href='/'),
    html.Br(),
    html.Div("Update", id="update-button", style={'display': 'none'}), #hidden div will update when page directed to

    html.Div(id='market-basket-load', style={'display': 'none'}), #intermediate market basket load - hidden div
    html.Div(id='perm-feature-wt-load', style={'display': 'none'}), #intermediate perm wt features - hidden div
    html.Div(id='perm-feature-wt-predummy-load', style={'display': 'none'}), #intermediate perm wt features pre dummy - hidden div
    html.Div(id='data-post-transform', style={'display': 'none'}), #intermediate data post transform - hidden div
    html.Div(id='best-model-load', style={'display': 'none'}), #intermediate best model metrics load - hidden div
    html.Div(id='cleansing-output-load', style={'display': 'none'}), #intermediate data cleansing log load - hidden div
    html.Div(id='rand-prediction-load', style={'display': 'none'}),
    html.Div(id='rand-prediction-single-row-load', style={'display': 'none'}),
    html.Div(id='fpr-load', style={'display': 'none'}),
    html.Div(id='tpr-load', style={'display': 'none'}),
    html.Div(id='time-series-date-col-load', style={'display': 'none'}), # date columns used for initial TS analysis - hidden div
#    html.Div(id='time-series-all-date-col-load', style={'display': 'none'}), # all date columns - hidden div
    html.Div(id='time-series-non-date-col-load', style={'display': 'none'}),
    html.Div(id='time-series-var-combos-load', style={'display': 'none'}),
    html.Div(id='time-series-trends-load', style={'display': 'none'}), #intermediate time series trend data - hidden div
    html.Div(id='time-series-forecast-load', style={'display': 'none'}), #intermediate time series forecast data - hidden div
    html.Div(id='time-series-best-models-load', style={'display': 'none'}), #intermediate time series best model data - hidden div
#    html.Div(id='rf-explanation', style={'display': 'none'}),
    html.Div(id='market-bar-click', style={'display': 'none'}),
    html.Div(id='market-scatter-click', style={'display': 'none'}),
    html.Div(id='features-metrics', style={'display': 'none'}),
    html.Div(id='features-metrics-predummy', style={'display': 'none'}),
    html.Div(id='confusion-matrix-load', style={'display': 'none'}),
    html.Div(id='dummy-variable-memory-load', style={'display': 'none'}),
    html.Div(id='features-collection-div', style={'display': 'none'}), #intermediary that collects 10 single instance values

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Business Insights', value='tab-1'),
        dcc.Tab(label='Market Basket Analysis', value='tab-2'),
        dcc.Tab(label='Time Series Analysis', value='tab-3', style={'display': 'block'}, id='time-series-container'),
        dcc.Tab(label='Single Instance Prediction', value='tab-4'),
        dcc.Tab(label='Optimal Model Report', value='tab-5'),
    ]),
    html.Div(id='tabs-content')
])

'''
Callbacks below. Callbacks provide tabs, but also provide internal state.
Hidden divs (above) hold data for sharing to other elements.
'''

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    best_model_data = pickle_load('./dashboard data/metrics_df')
    target_variable = best_model_data['target_variable']
    if tab == 'tab-1':
        # Feature importance table
        return html.Div([
            html.Div("Update tab 1", id="update-button-tab-1", style={'display': 'none'}), #hidden div will update when tab directed to

            #Feature importance bar graph on left, variable dropdown on right
            html.Div(className="row", children=[
                html.Div(className="six columns", children=[
                    dcc.Markdown('''
### Most Informative Attributes

*Most significant attributes influencing {target}*
                                 '''.format(target=target_variable)),
                    dcc.Graph(id = 'informative-feature-barchart',
                    )
                    ]),

                html.Div(className="six columns", children=[
                    dcc.Markdown('''
##### Pick Attribute for drilldown:
                                 '''),
                    dcc.Dropdown(id='feature-dropdown-component'),
                    html.Div(id='histogram-label'),
                    html.Div(id='click-label'),
                    dcc.Graph(id='feature-histogram')
                        ])]
                )
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.Div("Update tab 2", id="update-button-tab-2", style={'display': 'none'}),
            html.Br(),
            html.Div('No baskets were created because no categorical variables were found in the data.', id='no-baskets-warning', style={'display':'none'}),
            html.Div([ #market basket container, will be hidden if no categorical variables/baskets
                html.Div([html.Div(id='market-basket-summary', className='nine columns'),
                          html.Div(children=[
                            html.Div('See market baskets by:'),
                            html.Div(id='market-basket-added-col')
                            ], className="three columns"
                            )],
                         className='row'),
                html.Div(children=[
                    html.Div('Aggregated by:'),
                    dcc.Dropdown(id='market-basket-added-col-agg',
                            options=[{'label':x, 'value':x} for x in ['mean', 'sum', 'count']],
                            value='mean',
                        )], className="six columns", style={'display':'none'}
                    ),
                html.Div([
                    html.Div(
                        dcc.Graph(id='market-basket-scatter', clear_on_unhover=True, style={'height': 400}),
                            className='six columns'),
                    html.Div(
                        dcc.Graph(id='market-basket-bar', clear_on_unhover=True, style={'height': 400}),
                             className='six columns'),
                        ],
                    className='row'),
                html.Br(),
                html.Div(id='top-5-basket-tables'),
                html.Br(),
#                html.Details([html.Summary('Detailed Data'),
#                    html.Div(
#                        style={'overflowY': 'scroll', 'height': 500},
#                        id='market-basket-table'
#                        )])
                ], style={'display': 'block'}, id='market-basket-container')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Div("Update tab 3", id="update-button-tab-3", style={'display': 'none'}),
            html.Div('Time series analysis was skipped because no time series variables were detected in the data.', id='no-time-series-warning', style={'display':'none'}),
            html.Div('Time series analysis was skipped because the runtime setting "Ignore Time Series" = TRUE.', id='time-series-skipped', style={'display':'none'}),
            html.Div([ #time series container, will be hidden if no time series variables are present
                html.Div([ #time series graphs container, will be hidden if no time series analyses have run
                    html.Div([
                        html.Div([
                            html.Div(dcc.Markdown('#### Select time variable:')),
                            dcc.Dropdown(id='time-series-variable-col-dropdown')
                        ], className='three columns'),
                        html.Div([
                            html.Div(dcc.Markdown('#### Select feature(s):')),
                            dcc.Dropdown(id='time-series-analysis-col-dropdown', multi=True)
                        ], className='three columns'),
                        html.Div(id='time-series-best-model-summary', className='six columns'),
                        html.Div(id='feature-comparison-mode-summary', className='six columns')
                    ], className='row'),
                    html.Br(),
                    html.Div([
                        html.Div([
                            html.Div(id='time-series-predictions-line-description'),
                            dcc.Graph(id='time-series-predictions-line', clear_on_unhover=True)
                        ], className='six columns'),
                        html.Div([
                            html.Div(id='time-series-trend-line-description'),
                            dcc.Graph(id='time-series-trend-seasonality-line', clear_on_unhover=True)
                        ], className='six columns')
                    ], className='row'),
                    html.Br()
                ], style={'display': 'block'}, id='time-series-graphs-container'),
                html.Div(id='run-additional-time-series-container', children=[
                    html.Details([
                        html.Summary('Time variable or feature of interest not displayed? Run additional time series analyses.'),
                        html.Div(id='intermediate-time-series-run-data', style={'display': 'none'}),
                        html.Div([
                            html.Div([
                                html.Div(dcc.Markdown('Time Variable(s) for additional analysis:')),
                                dcc.Dropdown(id='run-time-series-variable-col-dropdown')
                            ], className='three columns'),
                            html.Div([
                                html.Div(dcc.Markdown('Feature(s) for additional analysis:')),
                                dcc.Dropdown(id='run-time-series-analysis-col-dropdown')
                            ], className='three columns')
                        ], className='row'),
                        html.Br(),
                        html.Button('Run Time Series Analysis', id='run-time-series-button', n_clicks=0),
                        dcc.ConfirmDialog(
                            id='run-time-series-confirmation',
                            message="Time Series finished running successfully.",
                            displayed=False,
                            submit_n_clicks=0
                        ),
                        html.Div(id='run-time-series-confirmation-output')
                    ])
                ])
            ])
        ])

    elif tab == 'tab-4':
        return html.Div([
                # Feature importance table
                html.Div("Update tab 4", id="update-button-tab-4", style={'display': 'none'}),
                html.Br(),

                html.Div([
                    html.Div([
                    html.Div(dcc.Graph(id='obs-proba-graph', style={'height': 400, 'width':525})),
                    html.Div('* One or more entered values are outside the min/max ranges observed in the data.',
                             style={'display':'none'},
                            id='outside-range-warning'),
                         ], className='six columns'),
                    html.Div([
                            html.Summary('Chosen Observation',
                                         style={'fontSize': 18, 'face': 'Arial'}),
                            html.Div([
                                    html.Div(id='rand-prediction-table')
                                    ], style={'overflowY': 'scroll', 'height': 300}),
#                            html.Div(
#                                    html.Button('Shuffle Random Observation', id='shuffle-observation-button'),
#                                    style={'marginTop':10}
#                                    ),
                            ], className='six columns', style={'marginTop':30}),
                        ], className='row', style={'height': 450}),

                #Slider that allows user to ascend/descend through data sample one at a time
                html.Div(id='pos-proba-slider-label', style={'fontSize': '19', 'face': 'Arial', 'width':700, 'margin': '0 auto', "textAlign": "center"}),
                html.Div('(slide to see different observations)', style={'fontSize': '15', 'face': 'Arial', 'width':700, 'margin': '0 auto', "textAlign": "center", 'fontStyle': 'italic'}),
                html.Div(dcc.Slider(id='pos-proba-slider', updatemode='drag'), style={'width':700, 'margin': '0 auto'}),
                html.Br(),
                html.Br(),

                #Auto-generated table with per-feature descriptive data
                html.Div(id='features-table', className="eight columns"),

                #10 divs for top-10 feature input boxes - each parent div is a row
                html.Div('Custom Value',
                    className="two columns",
                    style={'height': 40, 'fontSize': '19', 'face': 'Arial', 'fontWeight': 'bold'}),

                html.Div(
                        dcc.Input(id='feature-1-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-1-input-div'),
                html.Div(
                        dcc.Input(id='feature-2-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-2-input-div'),
                html.Div(
                        dcc.Input(id='feature-3-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-3-input-div'),
                html.Div(
                        dcc.Input(id='feature-4-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-4-input-div'),
                html.Div(
                        dcc.Input(id='feature-5-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-5-input-div'),
                html.Div(
                        dcc.Input(id='feature-6-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-6-input-div'),
                html.Div(
                        dcc.Input(id='feature-7-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-7-input-div'),
                html.Div(
                        dcc.Input(id='feature-8-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-8-input-div'),
                html.Div(
                        dcc.Input(id='feature-9-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-9-input-div'),
                html.Div(
                        dcc.Input(id='feature-10-input', type='number', min=0, value=0), #placeholder
                        className="two columns",
                        style={'height':40},
                        id='feature-10-input-div'),

        ])
    elif tab == 'tab-5':
        return html.Div([
        html.Div("Update tab 5", id="update-button-tab-5", style={'display': 'none'}),
    dcc.Markdown('### Data Cleansing Log'),
    html.Details(
        html.Div(
            style={'overflowY': 'scroll', 'height': 200, 'fontSize': '12'},
            id='cleansing-output'
            )
            ),
    dcc.Markdown('''
### Model Evaluation Statistics

*We ran a model on "held-out" observations to estimate model performance on unseen data.*

**Precision:** Of the data points that our model classified as positive, how many were truly positive?

**Recall:** Of the truly positive data points, how many did our model correctly classify as positive?

**Accuracy:** Of all truly positive and truly negative data points, how many did our model get correctly positive or negative?

**ROC Area Under Curve:** Represents our model's cumulative tradeoff between True Positive and False Positive given different classification thresholds.
100% AUC represents a perfect model that can provide a 100% True Positive Rate and a 0% False Positive Rate simultaneously.

                 '''),
    html.Div(id='model-metrics-table'),
    html.Div([
            dcc.Markdown('''
### Receiver Operating Characteristic

We can see a visualization of the trade-off between the true positive classification rate and the false positive classification rate with a ROC curve, generated from the performance of our model on the "held-out" dataset.

The black dotted line represents “luck”, or how we would expect a simple 50/50 coin flip would perform on this data.

The orange line represents our model, where the ultimate goal is to have the line flush against the top left corner; the top left corner represents perfection, with a 100% true positive rate and a 0% false positive rate.

The different points along the orange line represent spots that our model can occupy depending on the classification threshold we choose. If we are very sensitive to misclassifying a true label,
we may choose a point further to the right on the orange line, sacrificing and increasing the false positive rate in the process.
'''),
    html.Div(
        dcc.Graph(style={'height': 450, 'width': 650}, id='roc-curve')
            ),
    dcc.Markdown('''
### Confusion Matrix

Similar to the ROC Curve, the Confusion Matrix specifies where the model accurately classified positive and negative observations (diagonal),
but also where the model misclassifies positive observations as negative and vice versa (off-diagonal).

All observations fall into one of the four available buckets.
'''),
    html.Div(dcc.Graph(id='confusion-matrix', style={'width':'500'})),
])
        ])

def annotate(table):
    l=[
        {
            "x": "No",
            "y": "No",
            "font": {"color": "white", 'size':20},
            "showarrow": False,
            "text": table[0][0],
            "xref": "x1",
            "yref": "y1"
        },
        {
            "x": "No",
            "y": "Yes",
            "font": {"color": "white", 'size':20},
            "showarrow": False,
            "text": table[1][0],
            "xref": "x1",
            "yref": "y1"
        },
        {
            "x": "Yes",
            "y": "No",
            "font": {"color": "white", 'size':20},
            "showarrow": False,
            "text": table[0][1],
            "xref": "x1",
            "yref": "y1"
        },
        {
            "x": "Yes",
            "y": "Yes",
            "font": {"color": "white", 'size':20},
            "showarrow": False,
            "text": table[1][1],
            "xref": "x1",
            "yref": "y1"
        },
    ]
    return l

#load market basket data
@app.callback(Output('market-basket-load', 'children'),
              [Input('update-button-tab-2', 'children')])
def load_market_basket_data(n_clicks):
    market_basket = pd.read_csv('./dashboard data/market_basket.csv')
    return market_basket.to_json() #jsonify

#load feature importance data
@app.callback(Output('perm-feature-wt-load', 'children'),
              [Input('update-button-tab-1', 'children')])
def load_feature_importance_data(n_clicks):
#    if n_clicks > 0:
    perm_feature_wt = pd.DataFrame(pickle_load('./dashboard data/perm_feature_wt'))
    return perm_feature_wt.to_json()

#load feature importance *predummy* data
@app.callback(Output('perm-feature-wt-predummy-load', 'children'),
              [Input('update-button-tab-1', 'children')])
def load_feature_importance_predummy_data(n_clicks):
#    if n_clicks > 0:
    perm_feature_wt_predummy = pd.DataFrame(pickle_load('./dashboard data/perm_feature_wt_predummy'))
    return perm_feature_wt_predummy.to_json()

#load data cleansing output
@app.callback(Output('cleansing-output-load', 'children'),
              [Input('update-button-tab-5', 'children')])
def load_data_cleansing_output(n_clicks):
    data_cleansing_output = pickle_load('./dashboard data/data_cleansing_output')
    return json.dumps(data_cleansing_output)

#call in data cleansing output, then output to collapsible div on 5th tab
@app.callback(Output('cleansing-output', 'children'),
              [Input('cleansing-output-load', 'children')])
def update_data_cleansing_div(cleansing_output):
    data_cleansing_output = json.loads(cleansing_output)
    return dcc.Markdown(data_cleansing_output)

#load best model data
@app.callback(Output('best-model-load', 'children'),
              [Input('update-button', 'children')])
def load_best_model_data(n_clicks):
    best_model_data = pickle_load('./dashboard data/metrics_df') #this is a dictionary
    return json.dumps(best_model_data, default=str)

#load single row of randomly sampled prediction data
@app.callback(Output('rand-prediction-single-row-load', 'children'),
              [
               Input('update-button-tab-4', 'children'),
#               Input('shuffle-observation-button', 'n_clicks'),
               Input('pos-proba-slider', 'value')
               ])
def load_rand_prediction_data(tab_load, slider_value):
    rand_prediction_load = pd.DataFrame(pickle_load('./dashboard data/rand_prediction'))
    if slider_value == None:
        slider_value = 0
    rand_prediction_sample = rand_prediction_load.iloc[[slider_value],:]
    return rand_prediction_sample.to_json()

#load all rows of randomly sampled prediction data
@app.callback(Output('rand-prediction-load', 'children'),
              [Input('update-button-tab-4', 'children')])
def load_all_rand_prediction_data(tab_load):
    rand_prediction_load = pd.DataFrame(pickle_load('./dashboard data/rand_prediction'))
    return rand_prediction_load.to_json()

#load data features metrics (based on training data)
@app.callback(Output('features-metrics', 'children'),
              [Input('update-button-tab-4', 'children')])
def load_features_metrics_data(n_clicks):
    features_metrics = pd.DataFrame(pickle_load('./dashboard data/featureMetrics'))
    return features_metrics.to_json(orient='split') #keep indices/column order

#load data features metrics (based on training data) *predummy*
@app.callback(Output('features-metrics-predummy', 'children'),
              [Input('update-button-tab-4', 'children')])
def load_features_metrics_predummy_data(n_clicks):
    features_metrics_predummy = pd.DataFrame(pickle_load('./dashboard data/featureMetricsPreDummy'))
    return features_metrics_predummy.to_json(orient='split') #keep indices/column order

#load fpr data
@app.callback(Output('fpr-load', 'children'),
              [Input('update-button-tab-5', 'children')])
def load_fpr_data(n_clicks):
    fpr_load = pickle_load('./dashboard data/fpr')
    return json.dumps(fpr_load)

#load tpr data
@app.callback(Output('tpr-load', 'children'),
              [Input('update-button-tab-5', 'children')])
def load_tpr_data(n_clicks):
    tpr_load = pickle_load('./dashboard data/tpr')
    return json.dumps(tpr_load)

#load confusion matrix data
@app.callback(Output('confusion-matrix-load', 'children'),
              [Input('update-button-tab-5', 'children')])
def load_confusion_matrix_data(n_clicks):
    confusion_matrix_data = pd.DataFrame(pickle_load('./dashboard data/conf_matrix'))
    series = confusion_matrix_data.to_json(orient='split')
    return series

#load dummy variable memory data
@app.callback(Output('dummy-variable-memory-load', 'children'),
              [Input('update-button', 'children')])
def load_dummy_variable_memory_data(n_clicks):
    dummy_memory = pickle_load('./dashboard data/dummy_memory')
    dummy_memory = {x[:-3]:y for x,y in dummy_memory.items()} #remove ' = ' separator
    return json.dumps(dummy_memory)

#update confusion matrix
@app.callback(Output('confusion-matrix', 'figure'),
              [Input('update-button-tab-5', 'children'),
               Input('confusion-matrix-load', 'children'),
               Input('best-model-load', 'children')])
def update_confusion_matrix(load, confusion_matrix_data, model_metrics):
    table = pd.read_json(confusion_matrix_data, orient='split')
    table = table.values

    model_metrics = json.loads(model_metrics)
    target_variable = model_metrics['target_variable']

    return {"data": [
                {
                    "type": "heatmap",
                    "x": ['No', 'Yes'],
                    "y": ['Yes', 'No'],
                    "z": [[table[1][0], table[1][1]],
                          [table[0][0], table[0][1]]],
                    "showscale":False,
                   "colorscale":'Blues',
                   "reversescale": True,
                }
            ],
            "layout": {
                "xaxis": {"title": "Predicted "+target_variable},
                "yaxis": {"title": "True "+target_variable},
                "annotations": annotate(table),
                "margin":dict(t=35)
            }}

#load post model transform
@app.callback(Output('data-post-transform', 'children'),
              [Input('update-button', 'children')])
def load_data_post_transform(n_clicks):
#    if n_clicks > 0:
    data_post_transform = pd.read_csv('./dashboard data/data_post_transform.csv')
    return data_post_transform.to_json() #jsonify

#returns figure for ROC Curve graph
@app.callback(Output('roc-curve', 'figure'),
              [Input('best-model-load', 'children'),
               Input('fpr-load', 'children'),
               Input('tpr-load', 'children')])
def update_roc_curve_figure(model_metrics, fpr_load, tpr_load):
    model_metrics = json.loads(model_metrics)
    fpr_load = json.loads(fpr_load)
    tpr_load = json.loads(tpr_load)

    roc_auc = "{0:.0%}".format(model_metrics['roc_auc'])
    fpr = np.array(fpr_load)
    tpr = np.array(tpr_load)

    return go.Figure(data=[
            go.Scatter(x=fpr,
                       y=tpr,
                       mode='lines',
                       line=dict(color='darkorange', width=2),
                       name=f'ROC curve (area = {roc_auc})'),
            go.Scatter(x=[0, 1],
                       y=[0, 1],
                       mode='lines',
                       line=dict(color='navy',
                                 width=2, dash='dash'),
                       showlegend=False)
                    ],
            layout=go.Layout(xaxis=dict(title='False Positive Rate'),
                             yaxis=dict(
                                 title='True Positive Rate'),
                             margin=dict(t=15)))

#returns model metrics table
@app.callback(Output('model-metrics-table', 'children'),
              [Input('best-model-load', 'children')])
def update_model_metrics_table(model_metrics):
    model_metrics = json.loads(model_metrics)

    model_type = model_metrics['model_type']
    timestamp = model_metrics['timestamp'][:10]
    precision = "{0:.0%}".format(model_metrics['precision'])
    recall = "{0:.0%}".format(model_metrics['recall'])
    accuracy = "{0:.0%}".format(model_metrics['accuracy'])
    roc_auc = "{0:.0%}".format(model_metrics['roc_auc'])
    test_item_count = "{:,}".format(int(model_metrics['test_item_count']))

    col_names = [
    'Model Type',
    'Run Date',
    'Precision Score',
    'Recall Score',
    'Accuracy Score',
    'ROC Area Under Curve',
    'Validation Sample Count']

    formatted_metrics = [
        model_type,
        timestamp,
        precision,
        recall,
        accuracy,
        roc_auc,
        test_item_count]

    return generate_metrics_table(formatted_metrics, col_names)

#returns rand prediction table
@app.callback(Output('rand-prediction-table', 'children'),
              [Input('rand-prediction-single-row-load', 'children')])
def update_rand_prediction_table(rand_prediction):
    rand_prediction = pd.read_json(rand_prediction)
    rand_prediction = rand_prediction.transpose()
    rand_prediction['Feature'] = rand_prediction.index
    rand_prediction.columns = ['Value', 'Feature']
    rand_prediction = rand_prediction[['Feature', 'Value']]
    return generate_table_from_dataframe(rand_prediction)

#returns label for over histogram
@app.callback(Output('histogram-label', 'children'),
              [Input('feature-dropdown-component', 'value')])
def update_histogram_label(input1):
    return dcc.Markdown(
            '''
&nbsp;
*Values of **"{input1}"**:*
            '''.format(input1=input1)
            )

#returns histogram figure for histogram graph
@app.callback(Output('feature-histogram', 'figure'),
              [Input('feature-dropdown-component', 'value'),
               Input('data-post-transform', 'children'),
               Input('perm-feature-wt-load', 'children'),
               Input('best-model-load', 'children'),
               Input('dummy-variable-memory-load', 'children')
               ])
def update_histogram_feature(chosenVariable,
                             data_post_transform,
                             perm_feature_wt_data,
                             model_metrics,
                             dummy_memory
                             ):
    try:
        data_post_transform = pd.read_json(data_post_transform)
        dummy_memory = json.loads(dummy_memory)

        #for when first updating
        if chosenVariable == None:
            try:
                perm_feature_wt = pd.read_json(perm_feature_wt_data)
                perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
                perm_feature_list = list(perm_feature_wts['feature'])
                chosenVariable == perm_feature_list[0]
            except ValueError as e:
                print("perm_feature_wt", e)

        model_metrics = json.loads(model_metrics)
        target_variable = model_metrics['target_variable']
#May later want to rework the No and Yes to read the pos and neg class labels from ini

        if chosenVariable in dummy_memory.keys(): #Chosen variable is dummy
            dummyChildren = dummy_memory[chosenVariable]
            #PhoneService: ['PhoneService = No', 'PhoneService = Yes']

            #Calculate target_var distribution for No between all classes
            # and for Yes between all classes
            grouped = data_post_transform.groupby(target_variable).agg(
                {x:'sum' for x in dummyChildren})
            grouped['total'] = grouped.sum(axis=1)

            for c in dummyChildren:
                grouped[c] = grouped[c] / grouped['total']

            grouped = grouped.transpose()

            y0 = [] #numeric data
            y1 = []
            x = [] #labels
            for h in dummyChildren:
                y1.append(grouped[1][h])
                y0.append(grouped[0][h])
                x.append(h)

            data =[
                go.Bar(
                    x = x,
                    y = y0,
                    name = f'{target_variable}: No',
                    showlegend = True,
                    ),
                go.Bar(
                    x = x,
                    y = y1,
                    opacity = 0.75,
                    name = f'{target_variable}: Yes',
                    showlegend = True,
                    )
            ]
            feature_histogram = go.Figure(data=data,
                layout=go.Layout(margin=dict(t=15, b=300), barmode='overlay',
                    height=675))

        else: #Not dummy variable
            hist_x0 = go.Histogram(
                    x=data_post_transform[data_post_transform[target_variable]==0][chosenVariable],
                    name=f'{target_variable}: No',
                    opacity = 1,
                    histnorm='probability')
            hist_x1 = go.Histogram(
                    x=data_post_transform[data_post_transform[target_variable]==1][chosenVariable],
                    name=f'{target_variable}: Yes',
                    opacity = 0.75,
                    histnorm='probability')
            data = [hist_x0, hist_x1]
            feature_histogram = go.Figure(data=data,
                layout=go.Layout(margin=dict(t=15, b=300),
                barmode='overlay',
                height=675))
        return feature_histogram

    except ValueError as e:
        print(e)

#feed histogram dropdown OPTIONS
@app.callback(Output('feature-dropdown-component', 'options'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children'),
               Input('dummy-variable-memory-load', 'children'),
               Input('best-model-load', 'children')])
def update_histogram_feature_options(perm_feature_wt_data,
                                        data_post_transform,
                                        dummy_memory,
                                        model_metrics):

    #given post dummy col name (gender = Male) and dummy memory, returns pre dummy name (gender)
    def returnPreDummyCol(postDummyCol):
        for dummy_parent, dummy_children in dummy_memory.items():
            if postDummyCol in dummy_children:
                return dummy_parent
        return postDummyCol

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)
        dummy_memory = json.loads(dummy_memory)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
        perm_feature_list = list(perm_feature_wts['feature'])
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list

        #Convert all dummys to predummys
        all_feature_list = [returnPreDummyCol(x) for x in all_feature_list]
        all_feature_list = pd.Series(all_feature_list).drop_duplicates().tolist()

        #get target variable name and values
        model_metrics = json.loads(model_metrics)
        target_variable = model_metrics['target_variable']

        #Remove target variable option
        all_feature_list.remove(target_variable)

        options=[{'label':x, 'value':x} for x in all_feature_list]

        return options

    except ValueError as e:
        print(e)

#feed histogram dropdown VALUE
@app.callback(Output('feature-dropdown-component', 'value'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children'),
               Input('informative-feature-barchart', 'clickData'),
               Input('dummy-variable-memory-load', 'children')])
def update_histogram_feature_dropdown_value(perm_feature_wt_data,
                                             data_post_transform,
                                             chosenVariable,
                                             dummy_memory):

    #given post dummy col name (gender = Male) and dummy memory, returns pre dummy name (gender)
    def returnPreDummyCol(postDummyCol):
        for dummy_parent, dummy_children in dummy_memory.items():
            if postDummyCol in dummy_children:
                return dummy_parent
        return postDummyCol

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)
        dummy_memory = json.loads(dummy_memory)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
        perm_feature_list = list(perm_feature_wts['feature'])
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list

        dropdownValue = perm_feature_list[0]
        if chosenVariable != None:
            dropdownValue = chosenVariable['points'][0]['y']

        if ' = ' in dropdownValue:
            dropdownValue = returnPreDummyCol(dropdownValue)

        return dropdownValue

    except ValueError as e:
        print(e)

#feed features for feature bar graph
@app.callback(Output('informative-feature-barchart', 'figure'),
              [Input('perm-feature-wt-load', 'children')])
def update_bar_features(perm_feature_wt_data):
    perm_feature_wt = pd.read_json(perm_feature_wt_data)
    perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False).head(6)

    return go.Figure(data=[
            go.Bar(
                y=perm_feature_wts['feature'],
                x=perm_feature_wts['weight'],
                orientation = 'h'
                    )],
                layout=go.Layout(
                xaxis = dict(
                tickformat= ',.0%',
                title="Relative Importance"
                ),
                yaxis = dict(
                title="Attribute",
                autorange="reversed"
                ),
                margin=dict(t=15, l=205)

                ))

#feed market basket column dropdown values
@app.callback(Output('market-basket-added-col', 'children'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children')])
def update_market_basket_column_values(perm_feature_wt_data, data_post_transform):

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
        perm_feature_list = list(perm_feature_wts['feature'])
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list
        all_feature_list = [x for x in all_feature_list if " = " not in x]

        return dcc.Dropdown(options=[{'label':x, 'value':x} for x in all_feature_list],
                            value=perm_feature_list[0], id='market-basket-added-col-dropdown')
    except ValueError as e:
        print(e)

############### Market Basket Callbacks

@app.callback(Output('market-basket-scatter', 'figure'),
              [Input('market-basket-load', 'children'),
               Input('market-basket-added-col-dropdown', 'value'),
               Input('market-basket-added-col-agg', 'value'),
               Input('market-basket-bar', 'clickData'),
               Input('market-basket-scatter', 'clickData'),
               Input('market-bar-click', 'children'),
               Input('market-scatter-click', 'children')])
def update_market_basket_scatterplot(market_basket_data, col_name,  agg_method, barData, scatterData,
                                     barTs, scatterTs):
    market_basket = pd.read_json(market_basket_data)
    basket_col = col_name+'_basket_'+agg_method

    #check if any baskets exist (will be false if all variables are continuous)
    if len(market_basket) == 0:
        return None

    if barTs == None:
        barTs = 0
    if scatterTs == None:
        scatterTs = 0

    barBasket = None
    basketList = list(market_basket['Basket'])
    colorArray = list(market_basket[basket_col])

    if barTs > scatterTs and barData:
        barBasket = barData['points'][0]['x']
        #create array of colors here, where array length is # of unique baskets,
        #and highlight color is the # in the array equal to the place of
        #the basket chosen in scatterData
        chosenBasketNum = basketList.index(barBasket)
        colorArray[chosenBasketNum] = 'orange'
    if scatterTs > barTs and scatterData:
        scatterBasket = scatterData['points'][0]['text']
        chosenBasketNum = basketList.index(scatterBasket)
        colorArray[chosenBasketNum] = 'orange'

    #scale chosen dimension to be used for scatterplot sizes (avoids some overlaps)
    scaler = MinMaxScaler(feature_range=(15, 30))
    markerSize = list(scaler.fit_transform(market_basket[[basket_col]]).flatten())

    basketScatter = {'data': [
                        go.Scatter(
                            x=market_basket['Support'],
                            y=market_basket['Lift'],
                            text=basketList,
                            mode='markers',
#                            opacity=0.9,
                            marker=dict(
#                                size=22,
                                size=markerSize,
                                line={'width': 0.5, 'color': 'white'},
                                color=colorArray,
                                showscale=True,
                                colorbar={'title':col_name, 'titlefont':{'size':12}},
                                colorscale='Blues',
                                reversescale=True
                            ),
                            hoverinfo='none'
                        )
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Basket Frequency',
                               'titlefont': {'size': 18},
                               'showgrid': False,
                               'tickformat': ',.0%',
                               },
                        yaxis={'title': 'Basket Strength',
                               'titlefont': {'size': 18},
                               'showgrid': False,
                               },
                        hovermode='closest',
                        margin=dict(l=45, r=0, t=15, b=40),
                        shapes = [
                            {
                              "line": {
                                "color": "rgb(68, 68, 68)",
                                "dash": "solid",
                                "width": 1.5
                              },
                              "type": "line",
                              "x0": 0.5,
                              "x1": 0.5,
                              "xref": "paper",
                              "y0": 0,
                              "y1": 1,
                              "yref": "paper"
                            },
                            {
                              "line": {
                                "color": "rgb(68, 68, 68)",
                                "dash": "solid",
                                "width": 1.5
                              },
                              "type": "line",
                              "x0": 0,
                              "x1": 1,
                              "xref": "paper",
                              "y0": 0.5,
                              "y1": 0.5,
                              "yref": "paper"
                            }
                          ],
                        annotations=[
                            {
                              "x": 0,
                              "y": 0.95,
                              "font": {"size": 15, 'color': 'gray'},
                              "showarrow": False,
                              "text": "Stronger baskets<br>Less frequent",
                              "xref": "paper",
                              "yref": "paper",
                              "align": "left"
                            },
                            {
                              "x": 0,
                              "y": 0,
                              "font": {"size": 15, 'color': 'gray'},
                              "showarrow": False,
                              "text": "Weaker baskets<br>Less frequent",
                              "xref": "paper",
                              "yref": "paper",
                              "align": "left"
                            },
                            {
                              "x": 1,
                              "y": 0.95,
                              "font": {"size": 15, 'color': 'gray'},
                              "showarrow": False,
                              "text": "Stronger baskets<br>More frequent",
                              "xref": "paper",
                              "yref": "paper",
                              "align": "left"
                            },
                            {
                              "x": 1,
                              "y": 0,
                              "font": {"size": 15, 'color': 'gray'},
                              "showarrow": False,
                              "text": "Weaker baskets<br>More frequent",
                              "xref": "paper",
                              "yref": "paper",
                              "align": "left"
                            },
                          ],
                    )}

    return basketScatter

@app.callback(Output('market-basket-bar', 'figure'),
              [Input('market-basket-load', 'children'),
               Input('market-basket-added-col-dropdown', 'value'),
               Input('market-basket-added-col-agg', 'value'),
               Input('market-basket-scatter', 'clickData'),
               Input('market-basket-bar', 'clickData'),
               Input('market-bar-click', 'children'),
               Input('market-scatter-click', 'children')])
def update_market_basket_barchart(market_basket_data, col_name,  agg_method, scatterData, barData,
                                  barTs, scatterTs):
    market_basket = pd.read_json(market_basket_data)

    #check if any baskets exist (will be false if all variables are continuous)
    if len(market_basket) == 0:
        return None

    basket_col = col_name+'_basket_'+agg_method
    pop_col = col_name+'_pop_'+agg_method
    sorted_market_basket = market_basket.sort_values(by=basket_col, ascending=False)

    if barTs == None:
        barTs = 0
    if scatterTs == None:
        scatterTs = 0

    sortedBasketList = list(sorted_market_basket['Basket'])
    colorArray = ['#1662e0' for x in sortedBasketList]

    if barTs > scatterTs and barData:
        barBasket = barData['points'][0]['x']
        #create array of colors here, where array length is # of unique baskets,
        #and highlight color is the # in the array equal to the place of
        #the basket chosen in scatterData
        chosenBasketNum = sortedBasketList.index(barBasket)
        colorArray[chosenBasketNum] = 'orange'
    if scatterTs > barTs and scatterData:
        scatterBasket = scatterData['points'][0]['text']
        chosenBasketNum = sortedBasketList.index(scatterBasket)
        colorArray[chosenBasketNum] = 'orange'

    basketBar = {'data': [
                        go.Bar(
                            x=sortedBasketList,
                            y=sorted_market_basket[basket_col],
#                            text=sorted_market_basket['Basket'],
                            name=basket_col,
                            marker=dict(color=colorArray),
                            hoverinfo='none'
                        ),
                        go.Scatter(
                            y=sorted_market_basket[pop_col],
                            x=sortedBasketList,
                            name='population '+basket_col,
                            line=dict(
                                width=3,
                            ),
                            hoverinfo='none'
                        )
                    ],
                    'layout': go.Layout(
                        yaxis={'categoryorder':'category ascending',
                               'title': col_name,
                               'titlefont': {'size': 18},
                               'fixedrange':True},
                        xaxis={'showticklabels':False,
                               'title': 'Basket',
                               'titlefont': {'size': 18},
                               'fixedrange':True
                                },
                        hovermode='closest',
                        margin=dict(l=5, r=15, t=15, b=30),
                        legend=dict(x=-.1, y=1.2)
                    )}

    return basketBar

@app.callback(Output('market-bar-click', 'children'),
              [Input('market-basket-bar', 'clickData')])
def market_bar_clicked(barData):
    ts = datetime.now().timestamp()
#    print('bar clicked', ts)
    return ts

@app.callback(Output('market-scatter-click', 'children'),
              [Input('market-basket-scatter', 'clickData')])
def market_scatter_clicked(barData):
    ts = datetime.now().timestamp()
#    print('scatter clicked', ts)
    return ts

##### Create "Top 5" basket tables
@app.callback(Output('top-5-basket-tables', 'children'),
              [Input('market-basket-load', 'children'),
               Input('market-basket-added-col-dropdown', 'value'),
               Input('market-basket-added-col-agg', 'value')])
def update_market_basket_top5_tables(market_basket_data, col_name,  agg_method):
    market_basket = pd.read_json(market_basket_data)
    basket_col = col_name+'_basket_'+agg_method

    #check if any baskets exist (will be false if all variables are continuous)
    if len(market_basket) == 0:
        return None

    top5Lift = market_basket[['Basket', 'Lift']].sort_values(by='Lift', ascending=False).head(5)
    top5Lift['Lift'] = top5Lift['Lift'].map(lambda x: '{:.2f}x'.format(x))
    top5LiftTable = generate_table_from_dataframe(top5Lift)

    top5Support = market_basket[['Basket', 'Support']].sort_values(by='Support', ascending=False).head(5)
    top5Support['Support'] = top5Support['Support'].map(lambda x: '{:.0%}'.format(x))
    top5SupportTable = generate_table_from_dataframe(top5Support)

    top5Baskets = market_basket[['Basket', basket_col]].sort_values(by=basket_col, ascending=False).head(5)
    top5Baskets[basket_col] = top5Baskets[basket_col].map(lambda x: '{:.2f}'.format(x))
    top5BasketsTable = generate_table_from_dataframe(top5Baskets)

    bottom5Baskets = market_basket[['Basket', basket_col]].sort_values(by=basket_col, ascending=True).head(5)
    bottom5Baskets[basket_col] = bottom5Baskets[basket_col].map(lambda x: '{:.2f}'.format(x))
    bottom5BasketsTable = generate_table_from_dataframe(bottom5Baskets)

    def wrapTable(table, title):
        return html.Div([
                    html.Summary(title, style={'fontSize': '19', 'face': 'Arial', 'fontWeight': 'bold'}),
                    table
                ], className="six columns")

    return [html.Br(),
            html.Div([wrapTable(top5LiftTable, "Top 5 Baskets by Lift"),
                      wrapTable(top5SupportTable, "Top 5 Baskets by Support")], className="row"),
            html.Br(),
            html.Div([wrapTable(top5BasketsTable, "Top 5 Baskets by "+col_name),
                      wrapTable(bottom5BasketsTable, "Bottom 5 Baskets by "+col_name)], className="row")]

### Create summary basket line
@app.callback(Output('market-basket-summary', 'children'),
              [Input('market-basket-load', 'children'),
               Input('market-basket-added-col-dropdown', 'value'),
               Input('market-basket-added-col-agg', 'value'),
               Input('market-basket-scatter', 'clickData'),
               Input('market-basket-bar', 'clickData'),
               Input('market-bar-click', 'children'),
               Input('market-scatter-click', 'children')])
def update_market_basket_summary(market_basket_data, col_name,  agg_method, scatterData, barData,
                                 barTs, scatterTs):
    market_basket = pd.read_json(market_basket_data)
    basket_col = col_name+'_basket_'+agg_method

    if barTs == None:
        barTs = 0
    if scatterTs == None:
        scatterTs = 0

    basket = None
    if scatterTs > barTs and scatterData:
        basket = scatterData['points'][0]['text']
    elif barTs > scatterTs and barData:
        basket = barData['points'][0]['x']

    if basket:
        lift = '{:.2f}x'.format(market_basket[market_basket['Basket']==basket]['Lift'].values[0])
        support = '{:.0%}'.format(market_basket[market_basket['Basket']==basket]['Support'].values[0])
        basket_calc = '{:.2f}'.format(market_basket[market_basket['Basket']==basket][basket_col].values[0])
    else:
        basket = '(Click on graphs to see basket information.)'
        lift = ' '
        support = ' '
        basket_calc = ' '

    summary = [html.Div([
            html.Div([
                    html.Summary("Basket", style={'fontSize': '20', 'face': 'Arial', 'fontWeight': 'bold'}),
                    ], className='five columns'),
            html.Div([
                    html.Summary("Basket Strength", style={'fontSize': '14', 'face': 'Arial', 'fontWeight': 'bold'}),
                    ], className='two columns'),
            html.Div([
                    html.Summary("Basket Frequency", style={'fontSize': '14', 'face': 'Arial', 'fontWeight': 'bold'}),
                    ], className='two columns'),
            html.Div([
                    html.Summary(col_name+' '+agg_method, style={'fontSize': '14', 'face': 'Arial', 'fontWeight': 'bold', 'overflow':'hidden', 'whiteSpace': 'nowrap', 'textOverflow':'ellipsis'}),
                    ], className='three columns'),
        ], className="row"),
        html.Div([
            html.Div([
                    html.Summary(basket, style={'fontSize': '20', 'face': 'Arial'}),
                    ], className='five columns'),
            html.Div([
                    html.Summary(lift, style={'fontSize': '15', 'face': 'Arial'}),
                    ], className='two columns'),
            html.Div([
                    html.Summary(support, style={'fontSize': '15', 'face': 'Arial'}),
                    ], className='two columns'),
            html.Div([
                    html.Summary(basket_calc, style={'fontSize': '15', 'face': 'Arial'}),
                    ], className='three columns'),
        ], className="row", style={'overflowY': 'scroll', 'height': 110})
    ]

    return summary

#if no baskets/categorical variables, show warning, hide market basket elements
@app.callback(Output('no-baskets-warning', 'style'),
              [Input('market-basket-load', 'children')])
def show_no_baskets_warning(market_basket_data):
    market_basket = pd.read_json(market_basket_data)

    if len(market_basket) == 0:
        return {'marginLeft':20,
                'backgroundColor': '#ffea30',
                'fontSize': 14,
                'color': 'grey',
                'padding': '6px 6px 6px 12px',
                'border-radius': 10,
                'display':'block'}
    else:
        return {'display':'none'}

#if no baskets/categorical variables, show warning, hide market basket elements
@app.callback(Output('market-basket-container', 'style'),
              [Input('market-basket-load', 'children')])
def hide_market_basket_page_on_warning(market_basket_data):
    market_basket = pd.read_json(market_basket_data)

    if len(market_basket) == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

############### end market basket callbacks


######## Time Series analysis callbacks
@app.callback(Output('time-series-best-model-summary', 'children'),
              [Input('time-series-variable-col-dropdown', 'value'),
               Input('time-series-analysis-col-dropdown', 'value'),
               Input('time-series-best-models-load', 'children')])
def update_time_series_best_model_summary(time_col, col_name, ts_best_models):
    # if there is no time series data, return None
    # if zero/multiple analysis dropdown values are selected, return None
    if (ts_best_models == None) | (time_col == None) | (time_col == 'None') | (col_name == None) | (col_name == 'None') | (isinstance(col_name,(list,)) & (len(col_name) != 1)):
        return None
    else:
        # if col_name is a list, convert to string
        if isinstance(col_name,(list,)):
            col_name = ''.join(str(e) for e in col_name)

        # retrieve optimal model parameters
        ts_best_models = pd.read_json(ts_best_models)
        trend = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == col_name)].loc[:,['Trend']])
        damp = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == col_name)].loc[:,['Damped']])
        seas = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == col_name)].loc[:,['Seasonality']])
        periods = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == col_name)].loc[:,['Seasonal Periods']])

        if trend == 'add':
            trend_descr = 'Additive - overall amplitude of residuals is constant'
        elif trend == 'mul':
            trend_descr = 'Multiplicative - overall amplitude of residuals varies with the level of the underlying variable'
        else:
            trend_descr = 'None - no overall additive or multiplicative effects'

        if ((trend == 'add') | (trend == 'mul')) & (damp == True):
            damp_descr = 'Y'
        elif ((trend == 'add') | (trend == 'mul')) & (damp != True):
            damp_descr = 'N'
        else:
            damp_descr = 'N/A'

        if seas == 'add':
            seas_descr = 'Additive - amplitude of seasonal effects is constant'
        elif seas == 'mul':
            seas_descr = 'Multiplicative - amplitude of seasonal effects varies with the level of the underlying variable'
        else:
            seas_descr = 'None - no seasonal effects'

        if (seas == 'add') | (seas == 'mul'):
            periods_descr = periods
        else:
            periods_descr = 'N/A'

        summary = [html.Div(dcc.Markdown('#### Optimal Time Series Model:')),
            html.Div([
                html.Div([
                    html.Summary("Model Type:", style={'fontSize': '14', 'face': 'Arial', 'fontWeight': 'bold'})
                ], className= 'two columns'),
                html.Div([
                    html.Summary(trend_descr, style={'fontSize': '12', 'face': 'Arial'}),
                    ], className='six columns'),
                html.Div([
                    html.Summary("Damped Trend?:", style={'fontSize': '12', 'face': 'Arial', 'fontWeight': 'bold'})
                ], className= 'three columns'),
                html.Div([
                    html.Summary(damp_descr, style={'fontSize': '12', 'face': 'Arial'}),
                    ], className='one column')
            ], className="row"),
            html.Div([
                html.Div([
                    html.Summary("Seasonality:", style={'fontSize': '14', 'face': 'Arial', 'fontWeight': 'bold'})
                    ], className= 'two columns'),
                html.Div([
                    html.Summary(seas_descr, style={'fontSize': '12', 'face': 'Arial'}),
                    ], className='six columns'),
                html.Div([
                    html.Summary("Seasonal Periods:", style={'fontSize': '12', 'face': 'Arial', 'fontWeight': 'bold'})
                ], className= 'three columns'),
                html.Div([
                    html.Summary(periods_descr, style={'fontSize': '12', 'face': 'Arial'}),
                    ], className='one column'),
            ], className="row")
        ]
    return summary


@app.callback(Output('time-series-predictions-line-description', 'children'),
              [Input('time-series-forecast-load', 'children'),
               Input('time-series-analysis-col-dropdown', 'value')])
def update_ts_line_descr_output(forecast_data, col_name):
    #check if any time series variables exist
    if forecast_data == None:
        return dcc.Markdown('No Time Data')
    else:
        if isinstance(col_name,(list,)):
            if len(col_name) == 0:
                return dcc.Markdown('*No Feature Selected*')
            elif len(col_name) == 1:
                col_name = col_name[0]
            elif len(col_name) == 2:
                col_name = '"** and **"'.join(col_name)
            else:
                col_name = col_name[0]+'"**, **"'+col_name[1]+'"**, and **"'+col_name[2]
        return dcc.Markdown('*Predicted future values for Average **"{}"***'.format(col_name))

@app.callback(Output('time-series-trend-line-description', 'children'),
              [Input('time-series-forecast-load', 'children'),
               Input('time-series-analysis-col-dropdown', 'value')])
def update_ts_trend_descr_output(forecast_data, col_name):
    #check if any time series variables exist
    if forecast_data == None:
        return dcc.Markdown('No Time Data')
    else:
        if isinstance(col_name,(list,)):
            if len(col_name) == 0:
                return dcc.Markdown('*No Feature Selected*')
            elif len(col_name) == 1:
                col_name = col_name[0]
            elif len(col_name) == 2:
                col_name = '"** and **"'.join(col_name)
            else:
                col_name = col_name[0]+'"**, **"'+col_name[1]+'"**, and **"'+col_name[2]
        return dcc.Markdown('*History of Average **"{}"**, broken down into trend, seasonal, and residual components*'.format(col_name))

# Predictions Line Chart
@app.callback(Output('time-series-predictions-line', 'figure'),
              [Input('time-series-trends-load', 'children'),
               Input('time-series-forecast-load', 'children'),
               Input('time-series-variable-col-dropdown', 'value'),
               Input('time-series-analysis-col-dropdown', 'value'),
               Input('time-series-best-models-load', 'children')
               ])
def update_time_series_prediction_linechart(ts_trends_data, ts_forecast_data, time_col, col_name, ts_best_models):
    # if there is no time series data, do not display graph
    if (ts_trends_data == None) | (time_col == None) | (time_col == 'None') | (col_name == None) | (col_name == 'None'):
        return None
    else:
        ts_best_models = pd.read_json(ts_best_models)

        ts_trends = pd.read_json(ts_trends_data)
        ts_trends.sort_values('Datetime', inplace=True)
        ts_forecast = pd.read_json(ts_forecast_data)
        ts_forecast.sort_values('Datetime', inplace=True)

        if not isinstance(col_name,(list,)):
            col_name = [col_name]

        if len(col_name) == 0:
            return None
        else:
            for c in col_name:
                # get time trend and prediction for selected column
                add_or_mul = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == c)].loc[:,['Seasonality']])
                time_history = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Datetime']]
                col_history = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Original']]
                time_pred = ts_forecast[(ts_forecast['time variable'] == time_col) & (ts_forecast['feature'] == c)].loc[:,['Datetime']]
                col_pred = ts_forecast[(ts_forecast['time variable'] == time_col) & (ts_forecast['feature'] == c)].loc[:,['Forecast']]

                # stats
                #mean = np.mean(col_history.values)
                #std = np.std(col_history.values, ddof=1)
                #ucl = mean+3*std
                #lcl = mean-3*std
                #print('mean',c,':',mean)
                #print('std',c,':',std)
                #print('lcl',c,':',lcl)
                #print('ucl',c,':',ucl)

                if len(col_name) == 1:
                    hover_option = 'x+y'
                    hovermode_option = 'closest'
                    height_option = 450
                else:
                    hover_option = 'text+x+y'
                    hovermode_option = 'x'
                    height_option = 550

                if col_name.index(c) == 0:
                    color_option1 = 'rgb(31, 119, 180)'
                    color_option2 = 'rgb(255, 127, 14)'
                elif col_name.index(c) == 1:
                    color_option1 = 'rgb(98, 160, 203)'
                    color_option2 = 'rgb(255, 165, 86)'
                else:
                    color_option1 = 'rgb(145, 189, 219)'
                    color_option2 = 'rgb(255, 192, 137)'

                history = go.Scatter(
                                        x=np.squeeze(time_history, axis=1),
                                        y=np.squeeze(col_history, axis=1),
                                        name=c+' history',
                                        text=c,
                                        line=dict(
                                            color = color_option1,
                                            width=3
                                        ),
                                        hoverinfo=hover_option
                                    )
                predictions = go.Scatter(
                                            x=np.squeeze(time_pred, axis=1),
                                            y=np.squeeze(col_pred, axis=1),
                                            name='predicted '+c,
                                            text=c,
                                            line=dict(
                                                color = color_option2,
                                                width=3
                                            ),
                                            hoverinfo=hover_option
                                        )

                if col_name.index(c) == 0:
                    display_lines = [history, predictions]
                else:
                    display_lines.append(history)
                    display_lines.append(predictions)
            line = {'data': display_lines,
                    'layout': go.Layout(
                        yaxis={'title': 'Average value',
                               'titlefont': {'size': 18},
                               },
                        xaxis={'showticklabels':True,
                               'title': time_col,
                               'titlefont': {'size': 18},
                               'rangeslider': {'visible': True}
                                },
                        hovermode=hovermode_option,
                        legend=dict(x=0, y=1.4, orientation="h"),
                        height=height_option
                    )}
            return line

# Trend Seasonality Breakdown
@app.callback(Output('time-series-trend-seasonality-line', 'figure'),
              [Input('time-series-trends-load', 'children'),
               Input('time-series-variable-col-dropdown', 'value'),
               Input('time-series-analysis-col-dropdown', 'value'),
               Input('time-series-best-models-load', 'children')
               ])
def update_time_series_trend_seasonality_linecharts(ts_trends_data, time_col, col_name, ts_best_models):
    # if there is no time series data, do not display graph
    if (ts_trends_data == None) | (time_col == None) | (time_col == 'None') | (col_name == None) | (col_name == 'None'):
        return None
    else:
        ts_best_models = pd.read_json(ts_best_models)

        ts_trends = pd.read_json(ts_trends_data)
        ts_trends.sort_values('Datetime', inplace=True)

        if not isinstance(col_name,(list,)):
            col_name = [col_name]

        if len(col_name) == 0:
            return None
        else:
            for c in col_name:
                # get time trend for selected column
                add_or_mul = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == c)].loc[:,['Seasonality']])
                time_history = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Datetime']]
                col_history = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Original']]
                col_trend = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Trend']]
                col_seasonality = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Seasonality']]
                col_resid = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == c) & (ts_trends['type'] == add_or_mul)].loc[:,['Residual']]

                if len(col_name) == 1:
                    hover_option = 'x+y'
                    hovermode_option = 'closest'
                    height_option = 450
                    showlegend_option = True
                else:
                    hover_option = 'text+x+y'
                    hovermode_option = 'x'
                    height_option = 550
                    showlegend_option = False

                if col_name.index(c) == 0:
                    color_option1 = 'rgb(31, 119, 180)'
                    color_option2 = 'rgb(255, 127, 14)'
                    color_option3 = 'rgb(44, 160, 44)'
                    color_option4 = 'rgb(214, 39, 40)'
                elif col_name.index(c) == 1:
                    color_option1 = 'rgb(98, 160, 203)'
                    color_option2 = 'rgb(255, 165, 86)'
                    color_option3 = 'rgb(107, 189, 107)'
                    color_option4 = 'rgb(226, 104, 105)'
                else:
                    color_option1 = 'rgb(145, 189, 219)'
                    color_option2 = 'rgb(255, 192, 137)'
                    color_option3 = 'rgb(151, 209, 151)'
                    color_option4 = 'rgb(235, 149, 150)'

                chart1 = go.Scatter(
                    x=np.squeeze(time_history, axis=1),
                    y=np.squeeze(col_resid, axis=1),
                    name=c+' unexplained variance',
                    text=c,
                    line=dict(color = color_option4),
                    xaxis='x1',
                    yaxis='y1',
                    hoverinfo=hover_option,
                    showlegend=showlegend_option
                    )
                chart2 = go.Scatter(
                    x=np.squeeze(time_history, axis=1),
                    y=np.squeeze(col_seasonality, axis=1),
                    name=c+' seasonality',
                    text=c,
                    line=dict(color = color_option3),
                    xaxis='x2',
                    yaxis='y2',
                    hoverinfo=hover_option,
                    showlegend=showlegend_option
                    )
                chart3 = go.Scatter(
                    x=np.squeeze(time_history, axis=1),
                    y=np.squeeze(col_trend, axis=1),
                    name=c+' trend',
                    text=c,
                    line=dict(color = color_option2),
                    xaxis='x3',
                    yaxis='y3',
                    hoverinfo=hover_option,
                    showlegend=True
                    )
                chart4 = go.Scatter(
                    x=np.squeeze(time_history, axis=1),
                    y=np.squeeze(col_history, axis=1),
                    name=c+' observed',
                    text=c,
                    line=dict(color = color_option1),
                    xaxis='x4',
                    yaxis='y4',
                    hoverinfo=hover_option,
                    showlegend=True
                    )

                if col_name.index(c) == 0:
                    data = [chart4, chart3, chart2, chart1]
                else:
                    data.append(chart4)
                    data.append(chart3)
                    data.append(chart2)
                    data.append(chart1)

        layout = go.Layout(
            xaxis=dict(
                domain=[0, 1],
                title = time_col,
                titlefont= {'size': 18},
                showticklabels = False,
                fixedrange = True
            ),
            xaxis2=dict(
                domain=[0, 1],
                showticklabels = False,
                fixedrange = True
            ),
            xaxis3=dict(
                domain=[0, 1],
                showticklabels = False,
                fixedrange = True
            ),
            xaxis4=dict(
                domain=[0, 1],
                showticklabels = True,
                fixedrange = True
            ),
            yaxis=dict(
                domain=[0, 0.175],
                anchor='x1',
                fixedrange = True
            ),
            yaxis2=dict(
                domain=[0.275, 0.45],
                anchor='x2',
                fixedrange = True
            ),
            yaxis3=dict(
                domain=[0.55, 0.725],
                anchor='x3',
                fixedrange = True
            ),
            yaxis4=dict(
                domain=[0.825, 1],
                anchor='x4',
                fixedrange = True
            ),
            hovermode=hovermode_option,
            legend=dict(x=0, y=1.4, orientation="h"),
            height = height_option
        )

        fig = go.Figure(data=data, layout=layout)

        return fig

# if no time series data, show warning, hide time series elements
@app.callback(Output('no-time-series-warning', 'style'),
              [Input('time-series-date-col-load', 'children')])
def show_no_time_series_warning(ts_date_cols):
    ts_list = json.loads(ts_date_cols)
    if len(ts_list) == 0:
        return {'backgroundColor': '#ffea30',
                'fontSize': 14,
                'color': 'grey',
                'padding': '6px 6px 6px 12px',
                'border-radius': 10,
                'display':'block'}
    else:
        return {'display':'none'}

# if no time series variables, show warning, hide time series elements
@app.callback(Output('time-series-container', 'style'),
              [Input('time-series-date-col-load', 'children')])
def hide_time_series_page_on_warning(ts_date_cols):
    ts_list = json.loads(ts_date_cols)
    if len(ts_list) == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

# if there are time elements, but no time series runs have completed, display warning
@app.callback(Output('time-series-skipped', 'style'),
              [Input('time-series-var-combos-load', 'children'),
               Input('time-series-date-col-load', 'children')])
def show_time_series_skipped_warning(ts_runs, ts_date_cols):
    ts_list = json.loads(ts_date_cols)
    if (len(ts_list) > 0) & (ts_runs == None):
        return {'backgroundColor': '#ffea30',
                'fontSize': 14,
                'color': 'grey',
                'padding': '6px 6px 6px 12px',
                'border-radius': 10,
                'display':'block'}
    else:
        return {'display':'none'}

# if there are time elements, but no time series runs have completed, hide graphs
@app.callback(Output('time-series-graphs-container', 'style'),
              [Input('time-series-var-combos-load', 'children'),
               Input('time-series-date-col-load', 'children')])
def hide_time_series_graphs_on_warning(ts_runs, ts_date_cols):
    ts_list = json.loads(ts_date_cols)
    if (len(ts_list) > 0) & (ts_runs == None):
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(Output('feature-comparison-mode-summary', 'children'),
              [Input('time-series-best-models-load', 'children'),
               Input('time-series-variable-col-dropdown', 'value'),
               Input('time-series-analysis-col-dropdown', 'value')])
def update_feature_compare_options(ts_best_models, time_col, col_name):
    if isinstance(col_name,(list,)) & (len(col_name) > 1):
        block = [html.Div([
                        html.Div(dcc.Markdown('#### Feature Comparison Mode'), className='six columns'),
                        html.Div([
                            html.Br(),
                            html.Summary("Select up to 3 features to compare", style={'fontSize': '14', 'face': 'Arial'})
                        ], className= 'six columns')
                    ], className='row')
                ]
        return block
    else:
        return None

#if time series data analysis is null, show warning
@app.callback(Output('time-series-null-warning', 'style'),
              [Input('time-series-trends-load', 'children'),
               Input('time-series-variable-col-dropdown', 'value'),
               Input('time-series-analysis-col-dropdown', 'value')])
def show_null_time_series_warning(ts_trends_data, time_col, col_name):
    if (ts_trends_data == None) | (time_col == None) | (time_col == 'None') | (col_name == None) | (col_name == 'None'):
        return {'display':'none'}
    else:
        # time series trends
        ts_trends = pd.read_json(ts_trends_data)
        # Trend column
        col_trend = ts_trends[(ts_trends['time variable'] == time_col) & (ts_trends['feature'] == col_name) & (ts_trends['type'] == 'add')].loc[:,['Trend']]

        # if Trend column only contains nulls, display warning
        if col_trend.isnull().values.sum() == col_trend.shape[0]:
            return {'backgroundColor': '#ffea30',
                    'fontSize': 14,
                    'color': 'grey',
                    'padding': '6px 6px 6px 12px',
                    'border-radius': 10,
                    'display':'block'}
        else:
            return {'display':'none'}

# feed time series feature column dropdown OPTIONS
@app.callback(Output('time-series-analysis-col-dropdown', 'options'),
              [Input('time-series-var-combos-load', 'children'),
               Input('time-series-variable-col-dropdown', 'value'),
               Input('perm-feature-wt-load', 'children'),
               Input('time-series-analysis-col-dropdown', 'value'),
               Input('dummy-variable-memory-load', 'children')])
def update_time_series_feature_dropdown_options(ts_runs, time_col, perm_feature_wt_data, col_name, dummy_memory):
    # if no time series have run, return 'None'
    if ts_runs == None:
        return [{'label':x, 'value':x} for x in ['None']]

    # populate time variable options
    ts_runs = pd.read_json(ts_runs)
    if (time_col != 'None') & (time_col != None):
        non_ts_list = ts_runs[(ts_runs['time_var'] == time_col)].loc[:,['feature']].drop_duplicates().sort_values('feature').values.tolist()
    else:
        non_ts_list = ts_runs.loc[:,['feature']].drop_duplicates().sort_values('feature').values.tolist()
    if len(non_ts_list) > 1:
        non_ts_list = np.squeeze(non_ts_list, axis=1)

    # sort by feature weight
    perm_feature_wt = pd.read_json(perm_feature_wt_data)
    perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
    perm_feature_list = list(perm_feature_wts['feature'])

    # if at least one feature is selected, prioritize variables in the same category
    if not isinstance(col_name,(list,)):
        col_name = [col_name]
    same_cat_list = []
    dummy_memory = json.loads(dummy_memory)
    if len(col_name) > 0:
        for c in col_name:
            for dummy_parent, dummy_children in dummy_memory.items():
                if c in dummy_children:
                    additions = [x for x in dummy_children if ((x not in same_cat_list) and (x in non_ts_list))]
                    same_cat_list = same_cat_list + additions

    same_cat_perm_feature_list = [x for x in perm_feature_list if x in same_cat_list]
    all_perm_feature_list = [x for x in perm_feature_list if ((x in non_ts_list) and (x not in same_cat_list))]
    remaining_ts_list = [x for x in non_ts_list if x not in perm_feature_list]
    all_feature_list = same_cat_perm_feature_list + all_perm_feature_list + remaining_ts_list

    try:
        if len(all_feature_list) == 0:
            options=[{'label':x, 'value':x} for x in ['None']]
        elif len(all_feature_list)== 1:
            options=[{'label':x, 'value':x} for x in [all_feature_list]]
        else:
            if isinstance(col_name,(list,)):
                if len(col_name) >= 3:
                    options=[{'label':x, 'value':x, 'disabled':True} for x in all_feature_list]
                else:
                    options=[{'label':x, 'value':x} for x in all_feature_list]
            else:
                options=[{'label':x, 'value':x} for x in all_feature_list]
        return options
    except ValueError as e:
        print(e)

# feed time series feature column dropdown VALUE
@app.callback(Output('time-series-analysis-col-dropdown', 'value'),
              [Input('time-series-var-combos-load', 'children'),
               Input('perm-feature-wt-load', 'children')])
def update_time_series_feature_dropdown_value(ts_runs, perm_feature_wt_data):
    if ts_runs == None:
        return 'None'
    ts_runs = pd.read_json(ts_runs)
    non_ts_list = ts_runs['feature'].drop_duplicates().values.tolist()
    non_ts_list.sort()

    perm_feature_wt = pd.read_json(perm_feature_wt_data)
    perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=False)
    perm_feature_list = list(perm_feature_wts['feature'])

    all_perm_feature_list = [x for x in perm_feature_list if x in non_ts_list]
    remaining_ts_list = [x for x in non_ts_list if x not in perm_feature_list]
    all_feature_list = all_perm_feature_list + remaining_ts_list

    try:
        value=all_feature_list[0]
        return value
    except ValueError as e:
        print(e)

# feed time series variable column dropdown OPTIONS
@app.callback(Output('time-series-variable-col-dropdown', 'options'),
              [Input('time-series-var-combos-load', 'children'),
               Input('time-series-analysis-col-dropdown', 'value')])
def update_time_series_dropdown_options(ts_runs, feat_col):
    # if no time series have run, return 'None'
    if ts_runs == None:
        return [{'label':x, 'value':x} for x in ['None']]

    # populate time variable options
    ts_runs = pd.read_json(ts_runs)
    if (feat_col != 'None') & (feat_col != None):
        if not isinstance(feat_col,(list,)):
            feat_col = [feat_col]
        if len(feat_col) > 0:
            ts_list = ts_runs[(ts_runs['feature'] == feat_col[0])].loc[:,['time_var']].drop_duplicates().sort_values('time_var').values.tolist()
            for f in feat_col:
                ts_list_here = ts_runs[(ts_runs['feature'] == f)].loc[:,['time_var']].drop_duplicates().sort_values('time_var').values.tolist()
                ts_list = [x for x in ts_list if x in ts_list_here]
        else:
            ts_list = ts_runs.loc[:,['time_var']].drop_duplicates().sort_values('time_var').values.tolist()
    else:
        ts_list = ts_runs.loc[:,['time_var']].drop_duplicates().sort_values('time_var').values.tolist()
    ts_list = np.squeeze(ts_list, axis=1)

    try:
        if ts_list.size == 0:
            options=[{'label':x, 'value':x} for x in ['None']]
        elif ts_list.size == 1:
            options=[{'label':x, 'value':x} for x in [ts_list]]
        else:
            options=[{'label':x, 'value':x} for x in ts_list]
        return options
    except ValueError as e:
        print(e)

#feed time series feature column dropdown VALUE
@app.callback(Output('time-series-variable-col-dropdown', 'value'),
              [Input('time-series-var-combos-load', 'children')])
def update_time_series_dropdown_value(ts_runs):
    if ts_runs == None:
        return 'None'
    ts_runs = pd.read_json(ts_runs)
    ts_list = ts_runs['time_var'].drop_duplicates().values.tolist()
    ts_list.sort()

    try:
        if len(ts_list) == 0:
            value='None'
        else:
            value=ts_list[0]
        return value
    except ValueError as e:
        print(e)

#@app.callback(Output('add-or-mul', 'value'),
#              [Input('time-series-best-models-load', 'children'),
#               Input('time-series-variable-col-dropdown', 'value'),
#               Input('time-series-analysis-col-dropdown', 'value')])
#def update_model_radio_button(ts_best_models, time_col, col_name):
#    if ts_best_models == None:
#        return 'None'
#    else:
#        ts_best_models = pd.read_json(ts_best_models)
#        best_model = np.squeeze(ts_best_models[(ts_best_models['time variable'] == time_col) & (ts_best_models['feature'] == col_name)].loc[:,['Seasonality']])
#        return best_model

# load time series forecast dataset
@app.callback(Output('time-series-forecast-load', 'children'),
              [Input('run-time-series-confirmation', 'submit_n_clicks')])
def load_time_series_forecast(n_clicks):
    ts_list = pickle_load('./dashboard data/date_cols')
    #check if any time series variables exist
    if len(ts_list) == 0:
        return None
    else:
        ts_forecast = pickle_load('./dashboard data/ts_forecast')
        try:
            ts_forecast = ts_forecast.reset_index().to_json()
            return ts_forecast
        except:
            return None

# load time series trend dataset
@app.callback(Output('time-series-trends-load', 'children'),
              [Input('run-time-series-confirmation', 'submit_n_clicks')])
def load_time_series_trends(n_clicks):
    ts_list = pickle_load('./dashboard data/date_cols')
    #check if any time series variables exist
    if len(ts_list) == 0:
        return None
    else:
        ts_trends = pickle_load('./dashboard data/ts_trends')
        try:
            ts_trends = ts_trends.reset_index().to_json()
            return ts_trends
        except:
            return None

# load time series best model dataset
@app.callback(Output('time-series-best-models-load', 'children'),
              [Input('run-time-series-confirmation', 'submit_n_clicks')])
def load_time_series_best_models(n_clicks):
    ts_list = pickle_load('./dashboard data/date_cols')
    #check if any time series variables exist
    if len(ts_list) == 0:
        return None
    else:
        ts_best_model = pickle_load('./dashboard data/ts_best_model')
        try:
            ts_best_model = ts_best_model.reset_index().to_json()
            return ts_best_model
        except:
            return None

# load list of time analyses that have run
@app.callback(Output('time-series-var-combos-load', 'children'),
              [Input('run-time-series-confirmation', 'submit_n_clicks')])
def load_time_series_runs(n_clicks):
    #print('load_time_series_runs')
    ts_runs = pickle_load('./dashboard data/ts_runs')
    try:
        ts_runs = ts_runs.reset_index(drop=True).to_json()
    except:
        ts_runs = None
    return ts_runs

# one-off time-series analysis confirmation
@app.callback(Output('run-time-series-confirmation-output', 'children'),
              [Input('run-time-series-confirmation', 'submit_n_clicks')])
def update_time_series_finished_output(submit_n_clicks):
    if submit_n_clicks:
        return 'Time Series Analysis successful.'

# run one-off time-series analysis
@app.callback(Output('run-time-series-confirmation', 'displayed'),
              [Input('run-time-series-button', 'n_clicks')],
              [State('intermediate-time-series-run-data', 'children')])
def run_time_series_button_clicked(clicks, ts_run_options):
    if clicks > 0:
        time_var = ts_run_options[0]
        feature = ts_run_options[1]
        if Pages.time_series_analysis.runOneOffTimeSeries(time_var,feature) == True:
            print("Time Series Analysis successful.")
            return True
        else:
            print("There was an error running the time series analysis.")
            return False

# intermediate time series run data
@app.callback(Output('intermediate-time-series-run-data', 'children'),
              [Input('run-time-series-variable-col-dropdown', 'value'),
               Input('run-time-series-analysis-col-dropdown', 'value')])
def update_time_series_run_options(time_var, feature):
    return [time_var, feature]

# feed time series time variable column dropdown options for run TS button
@app.callback(Output('run-time-series-variable-col-dropdown', 'options'),
              [Input('time-series-date-col-load', 'children'),
               Input('time-series-var-combos-load', 'children'),
               Input('run-time-series-analysis-col-dropdown', 'value'),
               Input('update-button-tab-3', 'children')])
def update_time_series_to_run_time_variable_column_options(ts_date_cols, ts_runs, feature, clicks):
    #print('update_time_series_to_run_time_variable_column_options')
    ts_list_raw = json.loads(ts_date_cols)

    #print(feature)
    if (feature != 'all') & (feature != 'None') & (feature != None):
        try:
            ts_runs = pd.read_json(ts_runs)
            prev_run_list = ts_runs[(ts_runs['feature'] == feature)].loc[:,['time_var']].drop_duplicates().sort_values('time_var').values.tolist()
        except:
            prev_run_list = []
        prev_run_list = np.squeeze(prev_run_list, axis=1)
        if prev_run_list.size != 0:
            ts_list = [x for x in ts_list_raw if x not in prev_run_list]
        else:
            ts_list = ts_list_raw
    else:
        ts_list = ts_list_raw

    if len(ts_list) == 0:
        options=[{'label':x, 'value':x} for x in ['None']]
    elif len(ts_list) == 1:
        options=[{'label':x, 'value':x} for x in ts_list]
    else:
        options=[{'label':'All Time Variables', 'value':'all'}]
        ts_list.sort()
        for x in ts_list:
            options.append({'label':x, 'value':x})
    return options

# feed time series time variable column dropdown value for run TS button
@app.callback(Output('run-time-series-variable-col-dropdown', 'value'),
              [Input('time-series-date-col-load', 'children'),
               Input('update-button-tab-3', 'children')])
def update_time_series_to_run_time_variable_column_value(ts_date_cols, clicks):
    #print('update_time_series_to_run_time_variable_column_value')
    ts_list = json.loads(ts_date_cols)

    if len(ts_list) == 0:
        return 'None'
    elif len(ts_list) == 1:
        return ts_list[0]
    else:
        return 'all'

# feed time series feature variable column dropdown options for run TS button
@app.callback(Output('run-time-series-analysis-col-dropdown', 'options'),
              [Input('time-series-non-date-col-load', 'children'),
               Input('time-series-var-combos-load', 'children'),
               Input('run-time-series-variable-col-dropdown', 'value'),
               Input('update-button-tab-3', 'children')])
def update_time_series_to_run_feature_variable_column_options(ts_non_date_cols, ts_runs, time_col, clicks):
    #print('update_time_series_to_run_feature_variable_column_options')
    non_ts_list_raw = json.loads(ts_non_date_cols)

    #print(time_col)
    if (time_col != 'all') & (time_col != 'None') & (time_col != None):
        try:
            ts_runs = pd.read_json(ts_runs)
            prev_run_list = ts_runs[(ts_runs['time_var'] == time_col)].loc[:,['feature']].drop_duplicates().sort_values('feature').values.tolist()
        except:
            prev_run_list = []
        prev_run_list = np.squeeze(prev_run_list, axis=1)
        if prev_run_list.size != 0:
            non_ts_list = [x for x in non_ts_list_raw if x not in prev_run_list]
        else:
            non_ts_list = non_ts_list_raw
    else:
        non_ts_list = non_ts_list_raw

    if len(non_ts_list) == 0:
        options=[{'label':x, 'value':x} for x in ['None']]
    elif len(non_ts_list) == 1:
        options=[{'label':x, 'value':x} for x in non_ts_list]
    else:
        options=[{'label':'All Features', 'value':'all'}]
        non_ts_list.sort()
        for x in non_ts_list:
            options.append({'label':x, 'value':x})
    return options

# feed time series feature variable column dropdown value for run TS button
@app.callback(Output('run-time-series-analysis-col-dropdown', 'value'),
              [Input('time-series-non-date-col-load', 'children'),
               Input('update-button-tab-3', 'children')])
def update_time_series_to_run_feature_variable_column_value(ts_non_date_cols, clicks):
    #print('update_time_series_to_run_feature_variable_column_value')
    non_ts_list = json.loads(ts_non_date_cols)

    if len(non_ts_list) == 0:
        return 'None'
    elif len(non_ts_list) == 1:
        non_ts_list.sort()
        return non_ts_list[0]
    else:
        return 'all'

# load list of time variables
@app.callback(Output('time-series-date-col-load', 'children'),
              [Input('update-button', 'children')])
def load_time_series_variables(n_clicks):
    #print('load_time_series_variables')
    date_cols = pickle_load('./dashboard data/date_cols')
    return json.dumps(date_cols)

# load list of non-time variables
@app.callback(Output('time-series-non-date-col-load', 'children'),
              [Input('update-button', 'children')])
def load_non_time_series_variables(n_clicks):
    #print('load_non_time_series_variables')
    non_date_cols = pickle_load('./dashboard data/non_date_cols')
    return json.dumps(non_date_cols)


######## Sensitivity analysis callbacks

#create features table for sensitivity analysis
@app.callback(Output('features-table', 'children'),
              [Input('features-metrics-predummy', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_features_metrics_table(features_metrics, perm_feature_wt_data, dummy_memory):
    features_metrics = pd.read_json(features_metrics, orient='split')
    dummy_memory = json.loads(dummy_memory)

    perm_feature_wts = pd.read_json(perm_feature_wt_data) #load *predummy* data
    perm_feature_wts.sort_index(inplace=True)
    perm_feature_list = list(perm_feature_wts['feature'])

    #given post dummy col name (gender = Male) and dummy memory, returns pre dummy name (gender)
    def returnPreDummyCol(postDummyCol):
        for dummy_parent, dummy_children in dummy_memory.items():
            if postDummyCol in dummy_children:
                return dummy_parent
        return postDummyCol

    #sort by important features
    features_metrics.set_index(features_metrics['preDummy'], inplace=True) #apples-to-apples predummy-to-predummy
    features_metrics = features_metrics.loc[perm_feature_list, :].head(10)

    table_rows = []
    table_rows.append(html.Div(children = [
                        html.Div('Top Features', className = "seven columns", style={'fontSize': '19', 'face': 'Arial', 'fontWeight': 'bold'}),
                        html.Div('Min', className = "one column", style={'fontSize': '19', 'face': 'Arial', 'fontWeight': 'bold'}),
                        html.Div('Max', className = "one column", style={'fontSize': '19', 'face': 'Arial', 'fontWeight': 'bold'}),
                    ], className = "row", style={'height':40}))


    for i, x in features_metrics.iterrows():
        table_rows.append(
            html.Div(children = [
                        html.Div(x['preDummy'], className = "seven columns"),
                        html.Div(round(x['min'], 2), className = "one column"),
                        html.Div(round(x['max'], 2), className = "one column"),
                    ], className = "row", style={'height':40})
            )

    return table_rows

#If dataset has <10 fields, limit the amount of input fields that are shown
@app.callback(Output('feature-1-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_1(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 1: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-2-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_2(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 2: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-3-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_3(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 3: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-4-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_4(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 4: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-5-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_5(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 5: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-6-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_6(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 6: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-7-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_7(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 7: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-8-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_8(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 8: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-9-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_9(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 9: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}
@app.callback(Output('feature-10-input-div', 'style'),
              [Input('perm-feature-wt-predummy-load', 'children')])
def hide_input_10(perm_feature_wt_data):
    perm_feature_list = list(pd.read_json(perm_feature_wt_data)['feature'])
    if len(perm_feature_list) < 10: #if <X features, hide input box
        return {'display':'none'}
    else:
        return {'display':'block', 'height':40}

###Add features to slider

#Set proba slider min
@app.callback(Output('pos-proba-slider', 'min'),
              [Input('rand-prediction-load', 'children')])
def update_proba_slider_min(rand_predictions):
    rand_predictions = pd.read_json(rand_predictions)
    slider_min = rand_predictions.index.min()
    return slider_min
#Set proba slider max
@app.callback(Output('pos-proba-slider', 'max'),
              [Input('rand-prediction-load', 'children')])
def update_proba_slider_max(rand_predictions):
    rand_predictions = pd.read_json(rand_predictions)
    slider_max = rand_predictions.index.max()
    return slider_max
#Set proba slider marks
@app.callback(Output('pos-proba-slider', 'marks'),
              [Input('rand-prediction-load', 'children')])
def update_proba_slider_marks(rand_predictions):
    rand_predictions = pd.read_json(rand_predictions)
    slider_min = int(rand_predictions.index.min())
    slider_max = int(rand_predictions.index.max())
    min_proba = rand_predictions.loc[slider_min]['pos_proba']
    max_proba = rand_predictions.loc[slider_max]['pos_proba']
    marks = {
            slider_min: "{0:.0%}".format(min_proba),
            slider_max: "{0:.0%}".format(max_proba)
            }
#    marks = {slider_min:0, slider_max:199}
    return marks

#proba slider label pos-proba-slider-label
@app.callback(Output('pos-proba-slider-label', 'children'),
              [Input('best-model-load', 'children')])
def update_proba_slider_title(model_metrics):
    #get target variable name and values
    model_metrics = json.loads(model_metrics)
    target_variable = model_metrics['target_variable']
    return "Probability of "+target_variable

###

#Set input 1 to randomly chosen observation
@app.callback(Output('feature-1-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_1(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 1:
        top_feature = list(perm_feature_wts['feature'])[0] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-1-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-1-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 2 to randomly chosen observation
@app.callback(Output('feature-2-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_2(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 2:
        top_feature = list(perm_feature_wts['feature'])[1] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-2-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-2-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 3 to randomly chosen observation
@app.callback(Output('feature-3-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_3(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 3:
        top_feature = list(perm_feature_wts['feature'])[2] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-3-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-3-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 4 to randomly chosen observation
@app.callback(Output('feature-4-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_4(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 4:
        top_feature = list(perm_feature_wts['feature'])[3] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-4-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-4-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 5 to randomly chosen observation
@app.callback(Output('feature-5-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_5(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 5:
        top_feature = list(perm_feature_wts['feature'])[4] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-5-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-5-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox

#Set input 6 to randomly chosen observation
@app.callback(Output('feature-6-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_6(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 6:
        top_feature = list(perm_feature_wts['feature'])[5] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-6-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-6-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 7 to randomly chosen observation
@app.callback(Output('feature-7-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_7(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 7:
        top_feature = list(perm_feature_wts['feature'])[6] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-7-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-7-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 8 to randomly chosen observation
@app.callback(Output('feature-8-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_8(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 8:
        top_feature = list(perm_feature_wts['feature'])[7] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-8-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-8-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 9 to randomly chosen observation
@app.callback(Output('feature-9-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_9(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 9:
        top_feature = list(perm_feature_wts['feature'])[8] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-9-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-9-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox
#Set input 10 to randomly chosen observation
@app.callback(Output('feature-10-input-div', 'children'),
              [Input('rand-prediction-single-row-load', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_input_10(rand_prediction, perm_feature_wt_data, dummy_memory):
    rand_prediction = pd.read_json(rand_prediction) #get features of random prediction
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    if len(list(perm_feature_wts['feature'])) >= 10:
        top_feature = list(perm_feature_wts['feature'])[9] #find x most important feature
        top_feature_value = rand_prediction.iloc[0][top_feature]
    else: #not enough columns
        top_feature = ''
        top_feature_value = 0
    if isinstance(top_feature_value, (int, float, np.int64, np.float64)): #if value of top feature is number show input box
        inputBox = dcc.Input(id='feature-10-input', type='number', min=0, value=top_feature_value)
    else: #use dummy variable memory structure to get options for dummy dropdown
        dummy_memory = json.loads(dummy_memory)
        dummy_options = [x[x.find(' = ')+3:] for x in dummy_memory[top_feature]] #only keep post ' = ' text
        options=[{'label':x, 'value':x} for x in dummy_options]
        inputBox = dcc.Dropdown(id='feature-10-input', options=options, value=top_feature_value) #else is dummy - show dropdown
    return inputBox

#intermediary callback to collect 10 feature inputs and send to obs-proba-graph and outside range warning
@app.callback(Output('features-collection-div', 'children'),
              [Input('feature-1-input', 'value'),
               Input('feature-2-input', 'value'),
               Input('feature-3-input', 'value'),
               Input('feature-4-input', 'value'),
               Input('feature-5-input', 'value'),
               Input('feature-6-input', 'value'),
               Input('feature-7-input', 'value'),
               Input('feature-8-input', 'value'),
               Input('feature-9-input', 'value'),
               Input('feature-10-input', 'value')])
def collect_input_values(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10):
    collection = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    return json.dumps(collection)

#calculate model proba (when sliders change or when hit button, depending on performance)
@app.callback(Output('obs-proba-graph', 'figure'),
              [Input('features-collection-div', 'children'),
               Input('features-metrics-predummy', 'children'),
               Input('perm-feature-wt-predummy-load', 'children'),
               Input('best-model-load', 'children'),
               Input('rand-prediction-single-row-load', 'children'),
               Input('features-metrics', 'children'),
               Input('dummy-variable-memory-load', 'children')])
def update_features_values(feature_collection,
                           features_metrics, perm_feature_wt_data,
                           model_metrics, rand_prediction,
                           features_metrics_postdummy, dummy_memory):
    feature_collection = json.loads(feature_collection)
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = feature_collection

    features_metrics = pd.read_json(features_metrics, orient='split') #get pre-dummy order of X features
    features_metrics.set_index(features_metrics['preDummy'], inplace=True) #set features_metrics index to predummy names

    rand_prediction = pd.read_json(rand_prediction) #get features of rand prediction

    #set features_metrics values to those of the random observation
    rand_prediction_transpose = rand_prediction.transpose()
    rand_prediction_transpose.columns = ['value']
    features_metrics = features_metrics.join(rand_prediction_transpose, how='left')

    #get lookup of names of top 10 features
    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    perm_feature_list = list(perm_feature_wts['feature'])[:10]
    #for each element in perm_feature_list, wherever it is in features_metrics, set 'value' to input
    for name, value in zip(perm_feature_list, [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]):
        features_metrics.value[features_metrics.index==name] = value

    # get POST DUMMY features metrics (so that we can convert into proper form to get X probability from best_model)
    features_metrics_postdummy =  pd.read_json(features_metrics_postdummy, orient='split') #get post-dummy order of X features
    features_metrics_postdummy['value'] = 0
    dummy_memory = json.loads(dummy_memory)

    for row in features_metrics.itertuples(): #for each pre-dummy column
        newValues = preToPostDummyValueConversion(row.Index, row.value, dummy_memory) #find what post-dummy cols/vals should be
        for postDummyColumn, postDummyValue in newValues.items(): #update values in postdummy features_metrics
#            assert postDummyColumn in features_metrics_postdummy.index, "Dummy column "+postDummyColumn+" not found in post-dummified data."
            features_metrics_postdummy.value[features_metrics_postdummy.index==postDummyColumn] = postDummyValue

    #correctly ordered X to provide to get_proba_from_x
    X = features_metrics_postdummy['value'].values
    proba = get_proba_from_x([X])
    formatted_proba = ["{0:.0%}".format(x) for x in proba]

    #get target variable name and values
    model_metrics = json.loads(model_metrics)
    target_variable = model_metrics['target_variable']

    #Create bar chart figure
    bars = go.Figure(data=[go.Bar(
                            x=["No", "Yes"],
                            y=proba,
                            marker=dict(color=['#4186f4', '#f4b942']),
                            text=formatted_proba,
                            textposition = 'auto',
                        )],
                layout=go.Layout(
                    title='Probability of '+target_variable+' For Custom Observation',
                    xaxis = dict(title=target_variable),
                    yaxis = dict(title="Probability",
#                                 tickformat= ',.0%',
                                 fixedrange=True,
                                 range=[0, 1.05],
                                 showgrid = False,
                                 showticklabels = False),
#                    margin={'t': 0},
                    annotations=[
                        dict(
                            x=0.5,
                            y=1.05,
                            xref='x',
                            yref='y',
                            text='Shuffle/Enter Values Below to Observe Changes',
                            font=dict(color = "#cccccc", size = 12),
                            showarrow=False,
                        )
                    ]
                ))

    return bars

#Check if user input is outside range of previously seen data - if so, warn user
@app.callback(Output('outside-range-warning', 'style'),
              [Input('features-collection-div', 'children'),
               Input('features-metrics-predummy', 'children'),
               Input('perm-feature-wt-predummy-load', 'children')])
def warn_user_outside_range(feature_collection, features_metrics, perm_feature_wt_data):

    feature_collection = json.loads(feature_collection)
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = feature_collection
    features_metrics = pd.read_json(features_metrics, orient='split')

    perm_feature_wts = pd.read_json(perm_feature_wt_data)
    perm_feature_wts.sort_index(inplace=True)
    perm_feature_list = list(perm_feature_wts['feature'])

    #feature metrics (min, max) sorted by top 10 important features
    features_metrics = features_metrics.loc[perm_feature_list, :].head(10)
    featureValues = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    outOfRange = []
    for f_metrics, f_value in zip(features_metrics.iterrows(), featureValues):
        f_name = f_metrics[0]
        f_min = f_metrics[1]['min']
        f_max = f_metrics[1]['max']

        try:
            if not isinstance(f_value, str) and f_value != None and (f_value < f_min or f_value > f_max): #can not be string (in case of dummy)
                outOfRange.append(f_name)
        except TypeError as error:
            print(error)

#    print('out of range:', outOfRange) #if we want, can use outOfRange later to specify erring feature(s)

    if len(outOfRange) > 0: #a feature's custom input is outside min/max
        return {'marginLeft':20,
                'backgroundColor': '#ffea30',
                'fontSize': 14,
                'color': 'grey',
                'padding': '6px 6px 6px 12px',
                'border-radius': 10,
                'display':'block'}
    else: # all custom values within min/max range, hide warning
        return {'display':'none'}


#########
