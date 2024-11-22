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
