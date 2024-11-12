"""Ensemble.py"""
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from processing import preprocessed_df
import numpy as np
import pandas as pd
from processing import ProcessingMethods
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns



parent_directory = os.path.join("..", "ML_output")
os.makedirs(parent_directory, exist_ok=True)

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
    data.to_csv("processed.csv", index=False)
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

def end_to_end(filename, model):
    # Load the data
    df = pd.read_csv(filename)

    # Predict using the trained model
    y_pred = model.predict(df)

    # Add predictions to the dataframe
    df['PRCP_flag'] = y_pred

    df['Reviewed'] = 0
    df['Comment'] = ''

    # Save the dataframe to a new file
    file_path = os.path.join(parent_directory,"model_output.csv")
    df.to_csv(file_path, index=False)
    #df.to_json("model_output.json", orient="records", lines=True)
    file_path = os.path.join(parent_directory, "model_output.json")
    df.to_json(file_path, orient="records", lines=False)

    origin_file = pd.read_csv(r"C:\Users\cassa\Code\ML_output\preprocessed_df.csv") #raw input/with flag??

    # Compare the 'PRCP_flag' column in the two dataframes
    differences = df[df['PRCP_flag'] != origin_file['PRCP_flag']]

    # Save the differences to a new file
    #differences.to_csv("comparison_test.csv", index=False)
    file_path = os.path.join(parent_directory, "comparison_test.csv")
    differences.to_csv(file_path, index=False)

    return df


# Create the BalancedBaggingClassifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,  # number of models to train
    max_samples=0.5,  # undersample the majority class
    bootstrap=False,
    random_state=42,
    n_jobs=-1  # use all available CPU cores
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
mse = mean_squared_error(y_test, y_pred_bagging)
print("Mean Squared Error:", mse)

# class_counts = pd.DataFrame({'Class': y_train})
# plt.figure(figsize=(8, 6))
# sns.countplot(data=class_counts, x='Class', palette='Reds')

# # Set title and labels
# plt.title('Class Distribution (Class Imbalance)', fontsize=18)
# plt.xlabel('Class', fontsize=16)
# plt.ylabel('Count', fontsize=16)

# # Show the plot
# plt.show()
# Evaluate accuracy
# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred_bagging))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bagging))
# cm = confusion_matrix(y_test, y_pred_bagging)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()



end_to_end(r"C:\Users\cassa\Code\ML_output\preprocessed_df_no_flag.csv", bagging) #no flag


" Predictions for input file"

#processed_dataset = pd.read_csv(r)
origin_file = pd.read_csv(r"C:\Users\cassa\Code\ML_output\preprocessed_df.csv") #raw data?
processed_dataset = preprocess_input_data(origin_file)

# Split the new data into features
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

# Predict the PRCP_flag values
y_new_pred = bagging.predict(X_new)

print("Classification Report:\n", classification_report(y_new, y_new_pred))
# report = classification_report(y_new, y_new_pred, output_dict=True)

# # Convert classification report to a pandas DataFrame
# df_report = pd.DataFrame(report).transpose()

# # Plot Precision, Recall, and F1-Score
# df_report[['precision', 'recall', 'f1-score']].drop('accuracy').plot(kind='bar', figsize=(10,6))
# plt.title('Classification Report Metrics per Class')
# plt.ylabel('Score')
# plt.xticks(rotation=0)
# plt.show()

print("Confusion Matrix:\n", confusion_matrix(y_new, y_new_pred))
# cm = confusion_matrix(y_new, y_new_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3],annot_kws={"size": 16})
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()






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
#results_df.to_csv("prediction_actual.csv", index=False)
file_path = os.path.join(parent_directory, "prediction_actual.csv")
results_df.to_csv(file_path, index=False)

print("Actual vs Predicted results saved to actual_predicted.csv")




