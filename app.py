from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Define the InputTransformer class
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

# Define the OptimizedEnsemble class
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

# Load pipeline
base_dir = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(base_dir, "models", "optimized_ensemble_pipeline.pkl")

try:
    pipeline = joblib.load(pipeline_path)
    print(f"Pipeline loaded successfully from {pipeline_path}")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

@app.route('/')
def home():
    """
    Render the home page with input form.
    """
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    """
    Render the result page with the prediction.
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

        if not pipeline:
            raise Exception("Model pipeline is not loaded. Please check the pipeline file.")

        # Make prediction
        prediction_log = pipeline.predict(input_data)
        prediction = np.expm1(prediction_log)[0]  # Reverse log transformation

        return render_template('result.html', prediction=f"${prediction:,.2f}")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
