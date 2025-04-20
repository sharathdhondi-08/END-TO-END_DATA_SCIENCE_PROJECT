import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('customer_churn.csv')

# Drop 'customerID' column as it is not useful for prediction
data = data.drop(columns=['customerID'])

# Handle missing values if any (for simplicity, using median imputation for numerical columns)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce')
data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')
data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce')

# Fill missing values with the median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['MonthlyCharges'].fillna(data['MonthlyCharges'].median(), inplace=True)
data['tenure'].fillna(data['tenure'].median(), inplace=True)
data['SeniorCitizen'].fillna(data['SeniorCitizen'].median(), inplace=True)

# Separate the features (X) and target (y)
X = data.drop(columns=['Churn'])
y = data['Churn']

# Encode categorical variables
label_encoders = {}
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (RandomForestClassifier in this case)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional, can print out metrics like accuracy)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model and label encoders
with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump((model, label_encoders), f)

print("Model and label encoders saved successfully.")
