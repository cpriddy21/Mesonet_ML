""" processing.py """
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DatabaseConnection import DatabaseConnection

class ProcessingMethods:
    @staticmethod
    def describe_columns(df):
        pd.set_option('display.max_columns', None)
        numeric_columns = df.select_dtypes(include=['number'])
        print(numeric_columns.describe())
    @staticmethod
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
    @staticmethod
    def handle_category(df):
        # Label encode the CollectionMethod column
        le = LabelEncoder()
        df['CollectionMethod'] = le.fit_transform(df['CollectionMethod'])
    @staticmethod
    def drop_columns(df):
        # Drops unneeded columns that are still present after fully null columns are dropped
        bad_columns = ['BATV', 'DOOR', 'PMAS', 'PMAS_flag', 'PMAS_status', 'RGBV']
        df.drop(columns=bad_columns, inplace=True)
    @staticmethod
    def preprocess_data(connection):
        query = "SELECT * FROM sample"
        # Put table in a data frame
        df = pd.read_sql(query, con=connection)

        # Take out rows with missing values and fully null columns
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(inplace=True)

        # Convert datetime columns
        ProcessingMethods.handle_datetime(df)
        # Convert collection method
        ProcessingMethods.handle_category(df)
        # Drops remaining irrelevant columns
        ProcessingMethods.drop_columns(df)

        return df

class Process:
    @staticmethod
    def process_data():
        instance = DatabaseConnection.instance()
        connection = instance
        return ProcessingMethods.preprocess_data(connection)


preprocessed_df = Process.process_data()

