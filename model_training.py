""" This file does not work right now. Target column needs to be put in on lines 12 and 13, and i dont know what the
target column should be yet. Preprocess data also still has some issues like stuff still needs taken out i think """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example algorithm, replace with your choice
from sklearn.metrics import accuracy_score  # Example metric, replace with appropriate metrics

# Load preprocessed data from preprocessing file
from processing import preprocessed_df

# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])  # Replace 'target_column' with the name of your target variable
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train machine learning model
model = RandomForestClassifier()  # Replace with the appropriate algorithm and parameters
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate model performance4
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
