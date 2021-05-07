#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:41:19 2019

@author: ryanbasques
"""

import os

def get_client_data_filenames():
    assert os.path.exists('./client data'), "No 'client data' folder found."
    return os.listdir('./client data')