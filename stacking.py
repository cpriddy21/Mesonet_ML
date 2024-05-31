from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn import clone
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import class_weight
from processing import preprocessed_df
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class Weights
y_train = y_train.to_numpy()
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))

"*** Stacking ***"

# Define the base models
# Define your base classifier
base_clf = DecisionTreeClassifier()

# Define your resampling techniques
resampling_techniques = {
    'smote': SMOTE(),
    'undersample': RandomUnderSampler()
}

# Create an ensemble of classifiers, one for each resampling technique
classifiers = []
for name, resampler in resampling_techniques.items():
    # Resample the training data
    X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)

    # Train a classifier on the resampled data
    clf = clone(base_clf)
    clf.fit(X_resampled, y_resampled)

    # Add the classifier to the list
    classifiers.append((name, clf))

# Create a voting classifier from the ensemble
voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

# Train the voting classifier on the original data
voting_clf.fit(X_train_scaled, y_train)

# Use the voting classifier to make predictions
y_pred = voting_clf.predict(X_test_scaled)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\2011_input.csv")
# processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

# Standardize the features using the previously fitted scaler
X_new_scaled = scaler.transform(X_new)

# Predict the PRCP_flag values
y_new_pred = voting_clf.predict(X_new_scaled)

print("Classification Report:\n", classification_report(y_new, y_new_pred))
print("Confusion Matrix:\n", confusion_matrix(y_new, y_new_pred))