""" Still needs work. As of right now, it:
    - takes out the obvious irrelevant columns
    - takes out fully null columns
    - connects to mysql database (input ur own server info) """

import pandas as pd
import mysql.connector

''' Methods '''
# Prints ".describe() for all numeric columns
def describe_columns(df):
    pd.set_option('display.max_columns', None)
    numeric_columns = df.select_dtypes(include=['number'])
    print(numeric_columns.describe())

# Drops unneeded columns that are still present after fully null columns are dropped
def drop_columns(df):
    bad_columns = ['BATV', 'DOOR', 'PMAS', 'PMAS_flag', 'PMAS_status', 'RGBV']
    df.drop(columns=bad_columns, inplace=True)

def preprocess_data(connection):
    query = "SELECT * FROM sample"
    # Put table in a data frame
    df = pd.read_sql(query, con=connection)

    # Take out rows with missing values and fully null columns
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)

    # Drops remaining irrelevant columns
    drop_columns(df)

    return df


''' Connection To Database'''
# Connects to database that is set up in MySQL
connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="hgdaria14",
    database="Mesonet Data"
)

preprocessed_df = preprocess_data(connection)


