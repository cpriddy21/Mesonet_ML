from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


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


def end_to_end(filename, model):
    # Load the data
    df = pd.read_csv(filename)

    # Predict using the trained model
    y_pred = model.predict(df)

    # Add predictions to the dataframe
    df['PRCP_flag'] = y_pred

    # Save the dataframe to a new file
    df.to_csv("end_test_results.csv", index=False)

    origin_file = pd.read_csv(r"C:\Users\drm69402\Desktop\2012 rlly big test\2012_flag.csv")

    # Compare the 'PRCP_flag' column in the two dataframes
    differences = df[df['PRCP_flag'] != origin_file['PRCP_flag']]

    # Save the differences to a new file
    differences.to_csv("comparison_test.csv", index=False)

    return df


# Create the BalancedBaggingClassifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,  
    max_samples=0.5, 
    bootstrap=False,
    random_state=42,
    n_jobs=-1  
)

# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the bagging ensemble
bagging.fit(X_train, y_train)
# Predict using the bagging ensemble
y_pred_bagging = bagging.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred_bagging))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bagging))

end_to_end(r"C:\Users\drm69402\Desktop\2012 rlly big test\2012_no_flag.csv", bagging)
