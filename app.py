import os
import joblib
from flask import Flask, render_template, request
import numpy as np

# Flask app initialization
app = Flask(__name__)

# Determine the model path dynamically
base_dir = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(base_dir, "models", "optimized_ensemble_pipeline.pkl")

# Load the pipeline
pipeline = None
try:
    if os.path.exists(pipeline_path):
        print(f"Loading model from: {pipeline_path}")
        with open(pipeline_path, 'rb') as file:
            pipeline = joblib.load(file)
        print("Pipeline loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found at: {pipeline_path}")
except Exception as e:
    print(f"Error loading pipeline: {e}")

@app.route('/')
def home():
    """
    Render the main page with the input form.
    """
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    """
    Handle prediction requests and display results.
    """
    if not pipeline:
        return render_template('result.html', prediction="Error: Model pipeline is not loaded.")
    
    try:
        # Extract input data from the form
        data = request.form
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker_yes = int(data['smoker_yes'])
        region_southwest = int(data['region_southwest'])
        region_southeast = int(data['region_southeast'])
        bmi_smoker = bmi * smoker_yes  # Interaction feature

        # Prepare input for the model
        input_data = np.array([[age, bmi, children, smoker_yes, region_southwest, region_southeast, bmi_smoker]])

        # Make prediction
        prediction_log = pipeline.predict(input_data)
        prediction = np.expm1(prediction_log)[0]  # Reverse log transformation

        return render_template('result.html', prediction=f"${prediction:,.2f}")
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # Log current working directory and pipeline path for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Pipeline path: {pipeline_path}")

    # Confirm model existence
    if os.path.exists(pipeline_path):
        print("Model file exists.")
    else:
        print("Model file is missing.")

    app.run(debug=True)
