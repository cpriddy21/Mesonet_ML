"""Not Used"""
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from processing import preprocessed_df


over_sampler = RandomOverSampler(sampling_strategy='minority')  # automatically adjust to balance classes
under_sampler = RandomUnderSampler(sampling_strategy='majority')  # undersample majority classes equally

# Create the pipeline with both sampling strategies and the classifier
pipeline = Pipeline([
    ('over_sampling', over_sampler),
    ('under_sampling', under_sampler),
    ('classifier', BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,  # number of models to train
        bootstrap=True,
        random_state=42,
        n_jobs=-1  # use all available CPU cores
    ))
])

# Split data into features (X) and target variable (y)
X = preprocessed_df.drop(columns=['PRCP_flag'])
y = preprocessed_df['PRCP_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict using the pipeline
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict on new data
processed_dataset = pd.read_csv(r"C:\Users\drm69402\Desktop\input_file.csv")
X_new = processed_dataset.drop(columns=['PRCP_flag'])
y_new = processed_dataset['PRCP_flag']

y_new_pred = pipeline.predict(X_new)

print("Classification Report for new data:\n", classification_report(y_new, y_new_pred))
print("Confusion Matrix for new data:\n", confusion_matrix(y_new, y_new_pred))
