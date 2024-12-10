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

        # Drop original cols
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

    @staticmethod
    def preprocess_input_data(df):
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(inplace=True)
        # Handle datetime and category columns
        ProcessingMethods.handle_datetime(df)
        ProcessingMethods.handle_category(df)
        # Drop remaining irrelevant columns
        ProcessingMethods.drop_columns(df)
        return df

    @staticmethod
    def class_distribution(df):
        # Analyze class distribution
        class_distribution = df['PRCP_flag'].value_counts()
        print(class_distribution)

        # Visualize class distribution as a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=140)
        plt.title('Class Distribution (Pie Chart)')
        plt.show()