import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and label encoders
with open('model/churn_model.pkl', 'rb') as f:
    model, label_encoders = pickle.load(f)

# Get the list of features that the model expects
expected_features = model.feature_importances_.shape[0]

@app.route('/')
def home():
    return "Welcome to the Churn Prediction API! Use POST /predict to get results."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        print(f"Received data: {data}")

        # Convert the input data to a pandas DataFrame
        df = pd.DataFrame([data])

        # Drop 'customerID' if it exists (it's not useful for prediction)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        # Ensure that the columns in the input data match the model's expected columns
        # Add missing columns with default values (e.g., 'Unknown' for categorical or 0 for numerical)
        missing_columns = set(model.feature_names_in_) - set(df.columns)
        for col in missing_columns:
            if col in label_encoders:  # Handle categorical columns
                df[col] = 'Unknown'
            else:  # Handle numerical columns
                df[col] = 0  # For numerical columns, add 0 or appropriate default value

        # Ensure the correct order of columns
        df = df[model.feature_names_in_]

        # Encode categorical columns based on the label encoders from the training phase
        for col in df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                # Check for unseen categories and handle them
                df[col] = df[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'Unknown')
                # Add 'Unknown' to classes_ if it's not already present
                if 'Unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'Unknown')
                df[col] = label_encoders[col].transform(df[col])

        # Make prediction using the trained model
        prediction = model.predict(df)[0]

        # Return the result as JSON
        return jsonify({
            'prediction': 'Churn' if prediction == 1 else 'No Churn'
        })
    except Exception as e:
        # Handle any errors that occur during prediction
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
