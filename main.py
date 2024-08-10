from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

model = joblib.load('models/best_model.pkl')

REQUIRED_FEATURES = ["gender", "ever_married", "work_type", "residence_type", 
                     "smoking_status", "age", "bmi", "avg_glucose_level", 
                     "hypertension", "heart_disease"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Identify missing and extra features
        missing_features = [feature for feature in REQUIRED_FEATURES if feature not in data]
        extra_features = [feature for feature in data if feature not in REQUIRED_FEATURES]
        
        if missing_features or extra_features:
            return jsonify({
                "error": "Invalid input",
                "missing_features": missing_features,
                "extra_features": extra_features
            }), 400
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Predict using the model
        prediction = model.predict(input_data)[0]
        
        # Convert the prediction to a native Python type
        prediction = int(prediction)

        return jsonify({'prediction': prediction})

    except Exception as e:
        # Print the traceback to help with debugging
        print("Error occurred: ", str(e))
        print(traceback.format_exc())
        
        # Return a 500 error with the exception message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == '__main__':
  app.run(port=5000)