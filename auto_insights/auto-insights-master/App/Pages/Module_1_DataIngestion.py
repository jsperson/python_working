
# coding: utf-8

# In[12]:


"""
File to ingest multiple types of data and convert to pandas dataframe.

Supported inputs: csv path ('csv'), excel path ('excel'), sql query or database ('sql'), pickled data object ('pickle')
"""
import configparser
from sqlalchemy import create_engine

"""
File to ingest multiple types of data and convert to pandas dataframe.

Supported inputs: csv path ('csv'), excel path ('excel'), sql query or database ('sql'), pickled data object ('pickle')
"""
import pandas as pd

def generateDf():
    config = configparser.ConfigParser()
    config.read('./ml_box.ini')
    ingest_settings = config['INGEST']
    data_type = ingest_settings['datatype']



    if data_type == 'csv':
        print('Reading data type %s' % (data_type))
        data_source = ingest_settings['file_name']
        df = pd.read_csv(data_source, skip_blank_lines = True)
        return df

    elif data_type == 'excel':
        print('Reading data type %s' % (data_type))
        data_source = ingest_settings['file_name']
        df = pd.read_excel(data_source)
        return df

    elif data_type == 'sql':
        print('Reading data type %s' % (data_type))
        PROTOCOL = ingest_settings['PROTOCOL']
        DB_UNAME = ingest_settings['DB_UNAME']
        DB_PASS = ingest_settings['DB_PASS']
        DB_HOST = ingest_settings['DB_HOST']
        DB_PORT = ingest_settings['DB_PORT']
        DB_NAME = ingest_settings['DB_NAME']
        TABLE_NAME = ingest_settings['TABLE_NAME']
        engine_string = '%s://%s:%s@%s:%s/%s' % (PROTOCOL,DB_UNAME,DB_PASS,DB_HOST,DB_PORT,DB_NAME)
        query = 'select * from %s' % (TABLE_NAME)
        print(query)
        engine = create_engine(engine_string)
        df = pd.read_sql(query,engine)

        return df

    elif data_type == 'pickle':
        print('Reading data type %s' % (data_type))
        data_source = ingest_settings['file_name']
        df = pd.read_pickle(data_source)
        return df

    print('data_type specified not supported.\n')
    print("Please provide CSV path ('csv'), Excel path ('excel'), SQL connection ('sql'), or Pickled data ('pickle).")
