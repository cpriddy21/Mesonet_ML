""" neural_network.py"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from keras import Sequential
from keras.src.layers import Dense


def preprocess_input_data(data):
    # Take out rows with missing values and fully null columns
    data.dropna(axis=1, how='all', inplace=True)
    data.dropna(inplace=True)
    # Convert datetime columns
    ProcessingMethods.handle_datetime(data)
    # Convert collection method
    ProcessingMethods.handle_category(data)
    # Drops remaining irrelevant columns
    ProcessingMethods.drop_columns(data)

    return data

def calculate_confidence(predictions):
    # Calculate confidence as the absolute difference from 0.5
    confidences = np.abs(predictions - 0.5)
    # Normalize to range [0, 1]
    normalized_confidences = 1 - (2 * np.abs(confidences - 0.5))
    return normalized_confidences


# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.5)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

confidences = calculate_confidence(y_pred)

# Combine predictions and confidences
predictions_with_confidence = [(index, int(round(pred)), conf) for index, (pred, conf) in enumerate(zip(y_pred, confidences))]

# Print predictions with confidence and point
for index, pred, conf in predictions_with_confidence:
    print(f"Data Point {index}: Prediction: {pred}, Confidence: {conf:.2f}")

accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Evaluate the model: maybe change to MAE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse) 


input_dataset = pd.read_csv('/input_files/small.csv')
processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])

# Standardize the features using the previously fitted scaler
X_new_scaled = scaler.transform(X_new)

# Predict the PRCP_flag values
y_new_pred = model.predict(X_new_scaled).flatten()

# Round predictions to get binary values
y_new_pred_binary = [int(round(pred)) for pred in y_new_pred]

# Iterate over the new dataset and compare predictions with actual values
for index, (actual, predicted, conf) in enumerate(zip(processed_dataset['PRCP_flag'], y_new_pred_binary, confidences)):
    if actual != predicted and conf > 0.45:
        confidence_percent = conf * 100
        print(f"Data Point: {index}, Value in file: {actual}, Predicted: {predicted} CL: {confidence_percent:.2f}%")