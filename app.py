from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import logging
from utils.transformers import InputTransformer  # Import the InputTransformer class from utils.transformers
from models import OptimizedEnsemble  # Import the OptimizedEnsemble class from models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Print the current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Load pipeline
base_dir = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(base_dir, "models", "optimized_ensemble_pipeline.pkl")

logger.info(f"Base directory: {base_dir}")
logger.info(f"Attempting to load pipeline from {pipeline_path}")

pipeline = None
try:
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded successfully from {pipeline_path}")
    else:
        logger.error(f"Pipeline file not found at {pipeline_path}")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")

@app.route('/')
def home():
    """
    Render the home page with input form.
    """
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get form data
        data = request.form
        age = float(data['age'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker_yes = int(data['smoker_yes'])
        region_southwest = int(data['region_southwest'])
        region_southeast = int(data['region_southeast'])
        bmi_smoker = bmi * smoker_yes  # Interaction feature

        # Prepare input for the pipeline
        input_data = np.array([[age, bmi, children, smoker_yes, region_southwest, region_southeast, bmi_smoker]])

        if pipeline is None:
            raise Exception("Model pipeline is not loaded. Please check the pipeline file.")

        # Make prediction
        prediction_log = pipeline.predict(input_data)
        prediction = np.expm1(prediction_log)[0]  # Reverse log transformation

        logger.info(f"Prediction: {prediction}")
        return render_template('result.html', prediction=f"${prediction:,.2f}")
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
