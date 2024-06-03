"""Ensemble.py"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from sklearn.ensemble import BaggingClassifier, IsolationForest, GradientBoostingClassifier
import xgboost as xgb


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

model = xgb.XGBClassifier(
    objective='multi:softprob',  # for multi-class classification
    num_class=4,  # number of classes
    eval_metric='auc',  # evaluation metric
    seed=42,  # for reproducibility
    scale_pos_weight=1  # initial value
)

# Perform hyperparameter tuning
param_grid = {
    'scale_pos_weight': [1, 5, 10, 20]  # values to try
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Retrain your model using the best value of scale_pos_weight
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions with the tuned model
y_pred = best_model.predict(X_test)

# Evaluate the tuned model
print("Classification Report (Tuned Model):\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Tuned Model):\n", confusion_matrix(y_test, y_pred))

" Predictions for input file"

processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\2011_input.csv")
# processed_dataset = preprocess_input_data(input_dataset)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

# Standardize the features using the previously fitted scaler

# Predict the PRCP_flag values
y_new_pred = best_model.predict(X_new)


print("Classification Report:\n", classification_report(y_new, y_new_pred))
print("Confusion Matrix:\n", confusion_matrix(y_new, y_new_pred))


