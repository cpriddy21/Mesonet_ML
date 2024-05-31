"""Ensemble.py"""
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble import BaggingClassifier, IsolationForest, StackingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.tree import DecisionTreeClassifier


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


# Adjust predictions to be higher when in doubt
def adjust_predictions(y_pred, y_proba):
    for i in range(len(y_pred)):
        if y_pred[i] == 0 and y_proba[i][3] > 0.2:
            y_pred[i] = 3
        elif y_pred[i] == 1 and y_proba[i][2] > 0.2:
            y_pred[i] = 2
        elif y_pred[i] == 2 and y_proba[i][3] > 0.01:
            y_pred[i] = 3
    return y_pred


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

# Initialize the IsolationForest model
iso_forest = IsolationForest(contamination=0.1)

# Create the BaggingClassifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Define the stack
estimators = [
    ('Bagging Classifier', bagging),
    ('Isolation Forest', iso_forest)
]

# Initialize the StackingClassifier with a Logistic Regression as the final estimator
stacking_classifier = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

# Fit the stacking classifier to the training data
stacking_classifier.fit(X_train_scaled, y_train)

# Predict using the stacking classifier
y_pred_stacking = stacking_classifier.predict(X_test_scaled)

# Now you can evaluate the model using your preferred evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred_stacking))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stacking))







processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\2011_input.csv")
# processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

# Standardize the features using the previously fitted scaler
X_new_scaled = scaler.transform(X_new)

# Predict the PRCP_flag values
y_new_pred = stacking_classifier.predict(X_new_scaled)
y_new_proba = stacking_classifier.predict_proba(X_new_scaled)

# Adjust new predictions
y_new_pred_adjusted = adjust_predictions(y_new_pred.copy(), y_new_proba)


print("Classification Report:\n", classification_report(y_new, y_new_pred_adjusted))
print("Confusion Matrix:\n", confusion_matrix(y_new, y_new_pred_adjusted))