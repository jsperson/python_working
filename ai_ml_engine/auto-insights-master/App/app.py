#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:19:43 2019

@author: ryanbasques
"""

import dash

external_stylesheets = ['./assets/bWLwgP.css', 
#                        './assets/brPBPO.css',
                        './assets/spinnerLoading.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True