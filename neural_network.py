""" neural_network.py"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from Model import Model
from processing import preprocessed_df
import numpy as np

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

# Create model
model = Model.create_model('neural_network', X_train)
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.5)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

confidences = calculate_confidence(y_pred)

# Combine predictions and confidences
predictions_with_confidence = list(zip(y_pred_binary, confidences))

# Print predictions with confidence
for pred, conf in predictions_with_confidence:
    print(f"Prediction: {pred}, Confidence: {conf:.2f}")

accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
