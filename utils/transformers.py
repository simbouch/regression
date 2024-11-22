import numpy as np
import pandas as pd

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
