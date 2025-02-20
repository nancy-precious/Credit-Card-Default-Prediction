import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set visualization style
sns.set_style("whitegrid")

# Load the dataset
raw_data_path =  "C:/Users/Nancy/Documents/personal_projects/Credit_Card/data/raw/default of credit card clients.csv"
df = pd.read_csv(raw_data_path, encoding='latin1', header=1)  # Encoding specified to handle special characters

# Drop the ID column
df = df.drop(columns=['ID'])

# Separate features and target
X = df.drop(columns=["default payment next month"])  # Drop target variable
y = df["default payment next month"].values.ravel()  # Ensure y is 1D

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500, solver='saga', random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "notebooks/models/logistic_regression.pkl")

# Save the scaler
joblib.dump(scaler, "notebooks/models/scaler.pkl")

print("Model and scaler saved successfully.")
