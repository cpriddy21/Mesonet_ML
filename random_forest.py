"""Also runs. needs more statistical analysis"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example algorithm, replace with your choice
from sklearn.metrics import accuracy_score, mean_squared_error  # Example metric, replace with appropriate metrics

# Load preprocessed data from preprocessing file
from processing import preprocessed_df

# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP'])  # Replace 'target_column' with the name of your target variable
y = preprocessed_df['PRCP']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train machine learning model
model = RandomForestClassifier()  # Replace with the appropriate algorithm and parameters
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Model performance parameter
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Model performance parameter
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

