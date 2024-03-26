""" random_forest.py """
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from Model import Model
from processing import preprocessed_df

# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP'])
y = preprocessed_df['PRCP']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train machine learning model
model = Model.create_model('random_forest', X_train)
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Model performance parameter
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Model performance parameter
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

