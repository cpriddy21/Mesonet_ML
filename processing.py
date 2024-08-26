""" Process data from raw data file """
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DatabaseConnection import DatabaseConnection
import matplotlib.pyplot as plt


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
        df.drop(columns=['UTCTimestampCollected', 'LocalTimestampCollected', 'StandardTimestampCollected',
                         'UTCTimestampStored'], inplace=True)

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

    def class_distribution(df):
        # Analyze class distribution
        class_distribution = df['PRCP_flag'].value_counts()
        print(class_distribution)

        # Visualize class distribution as a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=140)
        plt.title('Class Distribution (Pie Chart)')
        plt.show()

    @staticmethod
    def preprocess_data(connection):
        query_2010 = "SELECT * FROM QA_KYMN_TBL_5min_2012 LIMIT 800000"
        # query_2009 = "SELECT * FROM QA_KYMN_TBL_5min_2009"

        # Put table in a data frame
        df = pd.read_sql(query_2010, con=connection)
        # df_2009 = pd.read_sql(query_2009, con=connection)
        # df = pd.concat([df_2009, df_2010], ignore_index=True)

        # Take out rows with missing values and fully null columns
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(inplace=True)

        # Convert datetime columns
        ProcessingMethods.handle_category(df)
        ProcessingMethods.handle_datetime(df)

        # Convert collection method
        ProcessingMethods.drop_columns(df)
        '''
        # Calculate sampling proportions based on raw data counts
        counts = {
            0: 90,
            1: 5,
            2: 3,
            3: 2
        }

        # Sample records from each class separately
        sampled_df = pd.DataFrame()
        for class_value, count in counts.items():
            class_df = df[df["PRCP_flag"] == class_value]
            sampled_class_df = class_df.sample(n=count, replace=True, random_state=42)
            sampled_df = pd.concat([sampled_df, sampled_class_df])'''

        df.to_csv('2012_800k.csv', index=False)
        return df


class Process:
    @staticmethod
    def process_data():
        instance = DatabaseConnection.instance()
        connection = instance
        # processed = ProcessingMethods.preprocess_data(connection)
        processed = pd.read_csv(r"C:\Users\drm69402\Desktop\2010_training_sorted.csv")
        return processed


preprocessed_df = Process.process_data()

