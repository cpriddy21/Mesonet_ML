""" neural_network.py"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
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
                Dense(4, activation="softmax")
            ])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=(['accuracy']))

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.5)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

input_dataset = pd.read_csv('/Users/daria/Downloads/PRCP_sample_20000 - Copy.csv')
processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])

# Standardize the features using the previously fitted scaler
X_new_scaled = scaler.transform(X_new)

# Predict the PRCP_flag values
y_new_pred = model.predict(X_new_scaled)
y_new_pred_classes = np.argmax(y_new_pred, axis=1)

print("starting")
var = 0
# Iterate over the new dataset and compare predictions with actual values
for index, (actual, predicted) in enumerate(zip(processed_dataset['PRCP_flag'], y_new_pred_classes)):
    if actual != predicted:
        print(f"{var} Data Point: {index}, Value in file: {actual}, Predicted: {predicted}")
        var = var + 1


# Cross-validation4
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for train_index, test_index in kfold.split(X_train_scaled, y_train):
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train.to_numpy()[train_index], y_train.to_numpy()[test_index]

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    y_pred_fold = model.predict(X_test_fold)
    y_pred_classes_fold = np.argmax(y_pred_fold, axis=1)
    fold_accuracy = accuracy_score(y_test_fold, y_pred_classes_fold)
    fold_accuracies.append(fold_accuracy)
    print("Fold Accuracy:", fold_accuracy)

print("Average Cross-Validation Accuracy:", np.mean(fold_accuracies))

