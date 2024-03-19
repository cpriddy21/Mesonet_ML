""" Still needs work. As of right now, it:
    - takes out the obvious irrelevant columns
    - takes out fully null columns
    - connects to mysql database (input ur own server info)
    - categorical value handling
    - datetime column handling """

import pandas as pd
import mysql.connector
from sklearn.preprocessing import LabelEncoder

''' Methods '''
# Prints ".describe() for all numeric columns. Does not preprocess. Just for initial analysis
def describe_columns(df):
    pd.set_option('display.max_columns', None)
    numeric_columns = df.select_dtypes(include=['number'])
    print(numeric_columns.describe())

# Converts datetime columns for ML alg
def handle_datetime(df):
    # UTCTimestamp col
    df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'])
    # Extract datetime features
    df['Year_UTC'] = df['UTCTimestampCollected'].dt.year
    df['Month_UTC'] = df['UTCTimestampCollected'].dt.month

    # LocalTimestamp col
    df['LocalTimestampCollected'] = pd.to_datetime(df['LocalTimestampCollected'])
    df['Year_Local'] = df['LocalTimestampCollected'].dt.year
    df['Month_Local'] = df['LocalTimestampCollected'].dt.month

    # StandardTimestamp col
    df['StandardTimestampCollected'] = pd.to_datetime(df['StandardTimestampCollected'])
    df['Year_Standard'] = df['StandardTimestampCollected'].dt.year
    df['Month_Standard'] = df['StandardTimestampCollected'].dt.month

    # UTCTimestampStored col
    df['UTCTimestampStored'] = pd.to_datetime(df['UTCTimestampStored'])
    df['Year_Stored'] = df['UTCTimestampStored'].dt.year
    df['Month_Stored'] = df['UTCTimestampStored'].dt.month

    # drop original cols
    df.drop(columns=['UTCTimestampCollected', 'LocalTimestampCollected', 'StandardTimestampCollected', 'UTCTimestampStored'], inplace=True)

# Handles Collection Method col
def handle_category(df):
    # Label encode the CollectionMethod column
    le = LabelEncoder()
    df['CollectionMethod'] = le.fit_transform(df['CollectionMethod'])

# Drops unneeded columns that are still present after fully null columns are dropped
def drop_columns(df):
    bad_columns = ['BATV', 'DOOR', 'PMAS', 'PMAS_flag', 'PMAS_status', 'RGBV']
    df.drop(columns=bad_columns, inplace=True)

# Where other methods are called and data is preprocessed
def preprocess_data(connection):
    query = "SELECT * FROM sample"
    # Put table in a data frame
    df = pd.read_sql(query, con=connection)

    # Take out rows with missing values and fully null columns
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)

    # Convert datetime columns
    handle_datetime(df)
    # Convert collection method
    handle_category(df)
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


