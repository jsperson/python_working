#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:22:11 2019

@author: ryanbasques
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import base64
import pickle
import dash_table
import json
import numpy as np
from app import app

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

def generate_table_from_dataframe(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

layout = html.Div([
    html.H1('AutoInsight Report'),
    dcc.Link('Go to data entry page', href='/'),
    html.Br(),
    html.Div("Update", id="update-button", style={'display': 'none'}), #hidden div will update when page directed to

    html.Div(id='market-basket-load', style={'display': 'none'}), #intermediate market basket load - hidden div
    html.Div(id='perm-feature-wt-load', style={'display': 'none'}), #intermediate perm wt features - hidden div
    html.Div(id='data-post-transform', style={'display': 'none'}), #intermediate data post transform - hidden div
    html.Div(id='best-model-load', style={'display': 'none'}), #intermediate best model metrics load - hidden div
    html.Div(id='rand-prediction-load', style={'display': 'none'}),
    html.Div(id='fpr-load', style={'display': 'none'}),
    html.Div(id='tpr-load', style={'display': 'none'}),
    html.Div(id='rf-explanation', style={'display': 'none'}),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Business Insights', value='tab-1'),
        dcc.Tab(label='Model Prediction', value='tab-2'),
        dcc.Tab(label='Optimal Model Report', value='tab-3'),
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
    if tab == 'tab-1':
        # Feature importance table
        return html.Div([
            html.Div("Update tab 1", id="update-button-tab-1", style={'display': 'none'}), #hidden div will update when tab directed to

            #Feature importance bar graph on left, variable dropdown on right
            html.Div(className="row", children=[
                html.Div(className="six columns", children=[
                    dcc.Markdown('''
### Optimal Model - Most Informative Features

*Which features most powerfully predict our target variable?*
                                 '''),
                    dcc.Graph(id = 'informative-feature-barchart',
                    )
                    ]),

                html.Div(className="six columns", children=[
                    dcc.Markdown('''
##### Pick feature for drilldown:
                                 '''),
#                    html.Div(id='feature-dropdown', children=
#                             dcc.Dropdown(id='feature-dropdown-component')),
                    dcc.Dropdown(id='feature-dropdown-component'),
                    html.Div(id='histogram-label'),
                    html.Div(id='click-label'),
                    dcc.Graph(id='feature-histogram')
                        ])]
                ),

            html.Div([

                dcc.Markdown('''

#### Top Market Baskets (scroll to see more)
##### *A basket contains fields that are often grouped in the data*

**Support:**  Percent of total observations containing the basket items; describes how frequently the basket occurs in the data.

**Lift:**  1.0x represents a random chance of the basket occuring. Lift >1.0x shows how much more often we see the basket than we would expect from random chance.
                    '''),

                html.Div(children=[

                    html.Div(children=[
                        html.Div('Define top market baskets by:'),

                        html.Div(id='market-basket-added-col')
#                        dcc.Dropdown(id='market-basket-added-col',
#                                options=[{'label':x, 'value':x} for x in all_feature_list],
#                                value=perm_feature_list[0],
#                            )
                        ], className="six columns"
                    ),

                    html.Div(children=[
                    html.Div('Aggregated by:'),

                    dcc.Dropdown(id='market-basket-added-col-agg',
                            options=[{'label':x, 'value':x} for x in ['mean', 'sum', 'count']],
                            value='mean',
                        )], className="six columns"
                    ),
                        ], style={'width':600, 'marginBottom':10}
                        , className="row"),

                html.Div(
                    style={'overflowY': 'scroll', 'height': 500},
                    id='market-basket-table'
                    ),
                    ])
        ])


    elif tab == 'tab-2':
        # Feature importance table
        return html.Div([
                html.Div("Update tab 2", id="update-button-tab-2", style={'display': 'none'}),

            html.Div([
                dcc.Markdown('''
##### Example Record

*Randomly chosen from dataset.*
                             '''),
                html.Div(id='rand-prediction-table'),
#                generate_table_from_dataframe(rand_prediction)
            ], style={'overflowX': 'scroll', 'width': 1200}),
            html.Div([
                dcc.Markdown('''
##### Example Record Feature Weighting and Explanation

*For this sample, how does each variable affect the final classification?*
                             '''),
                html.Div(id='rf-explanation-table'),
#                generate_table_from_dataframe(best_model['rf_explanation_example'])
            ])

        ])

    elif tab == 'tab-3':
        return html.Div([
                html.Div("Update tab 3", id="update-button-tab-3", style={'display': 'none'}),
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
            html.Div(
#                    [
#                    html.Img(
#                        src='data:image/png;base64,{}'.format(encoded_conf_matrix.decode()))
#                    ],
                    id='confusion-matrix-img')
        ]),
        ])

#update confusion matrix image
@app.callback(Output('confusion-matrix-img', 'children'),
              [Input('update-button-tab-3', 'children')])
def update_confusion_matrix(n_clicks):
#    if n_clicks > 0:
#        print('confusion matrix updated')
    conf_matrix_filename = './dashboard data/conf_matrix.png'
    encoded_conf_matrix = base64.b64encode(open(conf_matrix_filename, 'rb').read())
    return [html.Img(src='data:image/png;base64,{}'.format(encoded_conf_matrix.decode()))]

#load market basket data
@app.callback(Output('market-basket-load', 'children'),
              [Input('update-button-tab-1', 'children')])
def load_market_basket_data(n_clicks):
#    print("market basked data loaded")
#    if n_clicks > 0:
    market_basket = pd.read_csv('./dashboard data/market_basket.csv')
    return market_basket.to_json() #jsonify

#load feature importance data
@app.callback(Output('perm-feature-wt-load', 'children'),
              [Input('update-button-tab-1', 'children')])
def load_feature_importance_data(n_clicks):
#    if n_clicks > 0:
    perm_feature_wt = pd.DataFrame(pickle_load('./dashboard data/perm_feature_wt'))
    return perm_feature_wt.to_json()

#load best model data
@app.callback(Output('best-model-load', 'children'),
              [Input('update-button', 'children')])
def load_best_model_data(n_clicks):
    best_model_data = pickle_load('./dashboard data/metrics_df') #this is a dictionary
    return json.dumps(best_model_data, default=str)

#load rand prediction data
@app.callback(Output('rand-prediction-load', 'children'),
              [Input('update-button-tab-2', 'children')])
def load_rand_prediction_data(n_clicks):
    rand_prediction_load = pd.DataFrame(pickle_load('./dashboard data/rand_prediction'))
    return rand_prediction_load.to_json()

#load rf explanation data
@app.callback(Output('rf-explanation', 'children'),
              [Input('update-button-tab-2', 'children')])
def load_rf_explanation_data(n_clicks):
    rand_prediction_load = pd.DataFrame(pickle_load('./dashboard data/rf_explanation_example'))
    return rand_prediction_load.to_json()

#load fpr data
@app.callback(Output('fpr-load', 'children'),
              [Input('update-button-tab-3', 'children')])
def load_fpr_data(n_clicks):
    fpr_load = pickle_load('./dashboard data/fpr')
    return json.dumps(fpr_load)

#load tpr data
@app.callback(Output('tpr-load', 'children'),
              [Input('update-button-tab-3', 'children')])
def load_tpr_data(n_clicks):
    tpr_load = pickle_load('./dashboard data/tpr')
    return json.dumps(tpr_load)

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
              [Input('rand-prediction-load', 'children')])
def update_rand_prediction_table(rand_prediction):
    rand_prediction = pd.read_json(rand_prediction)
    return generate_table_from_dataframe(rand_prediction)

#returns rf explanation table
@app.callback(Output('rf-explanation-table', 'children'),
              [Input('rf-explanation', 'children')])
def update_rf_explanation_table(rf_explanation):
    rf_explanation = pd.read_json(rf_explanation)
    return generate_table_from_dataframe(rf_explanation)

#returns label for over histogram
@app.callback(Output('histogram-label', 'children'),
              [Input('feature-dropdown-component', 'value')])
def update_histogram_label(input1):
    return dcc.Markdown(
            '''
&nbsp;
*Histogram for values of **"{input1}"**:*
            '''.format(input1=input1)
            )

#returns histogram figure for histogram graph
@app.callback(Output('feature-histogram', 'figure'),
              [Input('feature-dropdown-component', 'value'),
               Input('data-post-transform', 'children'),
               Input('perm-feature-wt-load', 'children'),
               Input('best-model-load', 'children')
               ])
def update_histogram_feature(chosenVariable,
                             data_post_transform,
                             perm_feature_wt_data,
                             model_metrics
                             ):
    try:
        data_post_transform = pd.read_json(data_post_transform)

        #for when first updating
        if chosenVariable == None:
            try:
                perm_feature_wt = pd.read_json(perm_feature_wt_data)
                perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=True)
                perm_feature_list = list(perm_feature_wts['feature'])[::-1]
                chosenVariable == perm_feature_list[0]
            except ValueError as e:
                print("perm_feature_wt", e)

        model_metrics = json.loads(model_metrics)
        target_variable = model_metrics['target_variable']

        hist_x0 = go.Histogram(
                x=data_post_transform[data_post_transform[target_variable]==0][chosenVariable],
                name=f'{target_variable}: 0',
                opacity = 1,
                histnorm='probability')
        hist_x1 = go.Histogram(
                x=data_post_transform[data_post_transform[target_variable]==1][chosenVariable],
                name=f'{target_variable}: 1',
                opacity = 0.75,
                histnorm='probability')
        data = [hist_x0, hist_x1]
        feature_histogram = go.Figure(data=data, layout=go.Layout(margin=dict(t=15), barmode='overlay'))
        return feature_histogram

    except ValueError as e:
        print(e)

#feed histogram dropdown OPTIONS
@app.callback(Output('feature-dropdown-component', 'options'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children')])
def update_histogram_feature_options(perm_feature_wt_data,
                                             data_post_transform):

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=True)
        perm_feature_list = list(perm_feature_wts['feature'])[::-1]
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list

        options=[{'label':x, 'value':x} for x in all_feature_list]

        return options

    except ValueError as e:
        print(e)

#feed histogram dropdown VALUE
@app.callback(Output('feature-dropdown-component', 'value'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children'),
               Input('informative-feature-barchart', 'clickData')])
def update_histogram_feature_dropdown_value(perm_feature_wt_data,
                                             data_post_transform,
                                             chosenVariable):

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=True)
        perm_feature_list = list(perm_feature_wts['feature'])[::-1]
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list

        dropdownValue = perm_feature_list[0]
        if chosenVariable != None:
            dropdownValue = chosenVariable['points'][0]['y']

        return dropdownValue

    except ValueError as e:
        print(e)

#feed features for feature bar graph
@app.callback(Output('informative-feature-barchart', 'figure'),
              [Input('perm-feature-wt-load', 'children')])
def update_bar_features(perm_feature_wt_data):
    perm_feature_wt = pd.read_json(perm_feature_wt_data)
    perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=True)

    return go.Figure(data=[
            go.Bar(
                y=perm_feature_wts['feature'],
                x=perm_feature_wts['weight'],
                orientation = 'h'
                    )],
                layout=go.Layout(margin=dict(t=15, l=175)))

#feed market basket column dropdown values
@app.callback(Output('market-basket-added-col', 'children'),
              [Input('perm-feature-wt-load', 'children'),
               Input('data-post-transform', 'children')])
def update_market_basket_column_values(perm_feature_wt_data, data_post_transform):

    try:
        perm_feature_wt = pd.read_json(perm_feature_wt_data)
        data_post_transform = pd.read_json(data_post_transform)

        #list of all features
        perm_feature_wts = perm_feature_wt.sort_values('weight',ascending=True)
        perm_feature_list = list(perm_feature_wts['feature'])[::-1]
        all_feature_list = [x for x in data_post_transform.columns if x not in perm_feature_list]
        all_feature_list = perm_feature_list + all_feature_list

        return dcc.Dropdown(options=[{'label':x, 'value':x} for x in all_feature_list],
                            value=perm_feature_list[0], id='market-basket-added-col-dropdown')
    except ValueError as e:
        print(e)

#feed filtered market basket
@app.callback(Output('market-basket-table', 'children'),
              [Input('market-basket-added-col-dropdown', 'value'),
               Input('market-basket-added-col-agg', 'value'),
               Input('market-basket-load', 'children')
               ])
def update_market_basket_table_data(col_name, agg_method, market_basket_data):

    try:
        market_basket = pd.read_json(market_basket_data)

        basket_col = col_name+'_basket_'+agg_method
        pop_col = col_name+'_pop_'+agg_method
        m = market_basket[['Basket', 'Support', 'Lift', basket_col, pop_col]]

        d = dash_table.DataTable(
            id='market-basket-data',
            columns=[{"name": i, "id": i} for i in m.columns],
            data=m.to_dict("rows"),
            sorting=True,
            style_as_list_view=True,
            style_cell={'font': '14px arial',
                        'padding': '5px'},
            style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
            },
            style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left',
            } for c in ['Basket']
        ] + [
            {
                'if': {'column_id': c},
                'textAlign': 'center'
            } for c in ['Lift', 'Support']
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': 'rgb(242, 248, 255)'
            } for c in [basket_col, pop_col]
        ]
            )

        return d
    except ValueError as e: #no csv to start with
        print(e)
