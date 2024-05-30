""" neural_network.py"""
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from keras import Sequential
from keras.src.layers import Dense
import tensorflow as tf
from tensorflow.keras import backend as K


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


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


# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

' Imbalanced data techniques'
# SMOTE pretty bad, SMOTEENN really bad, AD whatever doesnt work at all

# random under: no class weights - 28499, pretty close on missed predictions
# rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
# X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# tomek: no class weights: 27348,
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)
y_resampled = y_resampled.to_numpy()

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
                Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(4, activation="softmax")
])

'''model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_resampled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(4, activation="softmax")
])'''

# Compile the model
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(class_weights), metrics=(['accuracy']))

# Train the model
model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, validation_split=0.5)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\input_file.csv")
# processed_dataset = preprocess_input_data(input_dataset)

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
