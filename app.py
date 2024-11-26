from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import logging
import importlib

# Define custom classes in the same script
class InputTransformer:
    """
    A custom transformer to ensure that input data is converted to a Pandas DataFrame
    with the correct column names before passing to the pipeline.
    """
    def __init__(self, column_names):
        self.column_names = column_names

    def transform(self, X, *_):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.column_names)
        return X

    def fit(self, X, y=None):
        return self

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

class OptimizedEnsemble:
    """
    Custom Optimized Ensemble Model combining Linear Regression and Gradient Boosting.
    """
    def __init__(self, weight_lr=0.6, weight_gb=0.4):
        self.weight_lr = weight_lr
        self.weight_gb = weight_gb
        self.lr_model = LinearRegression()
        self.gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)

    def fit(self, X, y):
        self.lr_model.fit(X, y)
        self.gb_model.fit(X, y)

    def predict(self, X):
        y_pred_lr = self.lr_model.predict(X)
        y_pred_gb = self.gb_model.predict(X)
        return self.weight_lr * y_pred_lr + self.weight_gb * y_pred_gb


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get base directory and set pipeline path
pipeline_path = "/tmp/8dd0df7c2a14461/models/optimized_ensemble_pipeline.pkl"

logger.info(f"Pipeline path: {pipeline_path}")

# Load pipeline
pipeline = None
try:
    if os.path.exists(pipeline_path):
        logger.info("Pipeline file exists. Attempting to load...")
        with open(pipeline_path, "rb") as f:
            module = importlib.import_module("__main__")
            setattr(module, "InputTransformer", InputTransformer)
            setattr(module, "OptimizedEnsemble", OptimizedEnsemble)
            pipeline = joblib.load(f)
        logger.info(f"Pipeline loaded successfully from {pipeline_path}")
    else:
        logger.error(f"Pipeline file not found at {pipeline_path}")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")


@app.route('/')
def home():
    """
    Render the home page with the input form.
    """
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    """
    Handle prediction and display results.
    """
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

        # Check if the pipeline is loaded
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
    app.run(debug=True, host="0.0.0.0")
