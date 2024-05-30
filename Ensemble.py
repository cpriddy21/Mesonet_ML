"""Ensemble.py"""
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from keras import Sequential
from keras.src.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.ensemble import BaggingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.tree import DecisionTreeClassifier


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


def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def predict(self, X):
    # Make predictions using all the base classifiers
    predictions = [classifier.predict(X) for classifier in self.classifiers]
    # Aggregate predictions using majority voting
    majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    return majority_votes


# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Wrap the Keras model to be used with scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Create the BalancedBaggingClassifier
bagging = BaggingClassifier(
    estimator=None,
    n_estimators=10,  # number of models to train
    max_samples=0.5,  # undersampling the majority class to balance
    bootstrap=True,
    random_state=42,
    n_jobs=-1  # use all available CPU cores
)

# Train the bagging ensemble
bagging.fit(X_train_scaled, y_train)

# Predict using the bagging ensemble
y_pred_bagging = bagging.predict(X_test_scaled)

# Evaluate accuracy
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print("Accuracy of Bagging Ensemble:", accuracy_bagging)

processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\input_file.csv")
# processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

# Standardize the features using the previously fitted scaler
X_new_scaled = scaler.transform(X_new)

# Predict the PRCP_flag values
y_new_pred = bagging.predict(X_new_scaled)

print("Classification Report:\n", classification_report(y_new, y_new_pred))
print("Confusion Matrix:\n", confusion_matrix(y_new, y_new_pred))

print("starting")
var = 0
# Iterate over the new dataset and compare predictions with actual values
for index, (actual, predicted) in enumerate(zip(processed_dataset['PRCP_flag'], y_new_pred)):
    if actual != predicted:
        print(f"{var} Data Point: {index}, Value in file: {actual}, Predicted: {predicted}")
        var = var + 1

results_df = pd.DataFrame({
    'Actual_PRCP_flag': y_new,
    'Predicted_PRCP_flag': y_new_pred
})

# Save the results DataFrame to a CSV file
results_df.to_csv(r"C:\Users\drm69402\Desktop\actual_predicted.csv", index=False)
print("Actual vs Predicted results saved to actual_predicted.csv")

