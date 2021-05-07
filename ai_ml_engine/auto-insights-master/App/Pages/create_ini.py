#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:16:17 2019

@author: ryanbasques
"""

import configparser
import ast

def create_ini(options_dict):

    options_dict = ast.literal_eval(options_dict)

    Config = configparser.ConfigParser()
    cfgfile = open("./ml_box.ini",'w')

    # add the settings to the structure of the file
    Config.add_section('INGEST')
    Config.set('INGEST','datatype', options_dict['datatype'])
    Config.set('INGEST','protocol', options_dict['protocol'])
    Config.set('INGEST','db_uname', options_dict['db_uname'])
    Config.set('INGEST','db_pass', options_dict['db_pass'])
    Config.set('INGEST','db_host', options_dict['db_host'])
    Config.set('INGEST','db_port', options_dict['db_port'])
    Config.set('INGEST','db_name', options_dict['db_name'])
    Config.set('INGEST','table_name', options_dict['table_name'])
    Config.set('INGEST','file_name', options_dict['file_name'])

    Config.add_section('RUNTIME')
    Config.set('RUNTIME','dash_data_path', options_dict['dash_data_path'])
    Config.set('RUNTIME','na_handling', options_dict['na_handling'])
    Config.set('RUNTIME','training', options_dict['training'])
    Config.set('RUNTIME','label_field', options_dict['label_field'])
    Config.set('RUNTIME','pos_label', options_dict['pos_label'])
    Config.set('RUNTIME','neg_label', options_dict['neg_label'])
    Config.set('RUNTIME','ts_ignore', options_dict['ts_ignore'])
    Config.set('RUNTIME','mb_ignore', options_dict['mb_ignore'])
    Config.set('RUNTIME','scoring_metric', options_dict['scoring_metric'])
    Config.set('RUNTIME','oversample', options_dict['oversample'])

    Config.write(cfgfile)
    cfgfile.close()

    return True
