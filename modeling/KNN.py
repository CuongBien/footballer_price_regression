import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class KNNModel:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self
    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.pipeline.predict(X)
    def score(self, X_test, y_test, metric="rmse"):
        y_pred = self.predict(X_test)

        if metric == "rmse":
            return np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == "mse":
            return mean_squared_error(y_test, y_pred)
        elif metric == "r2":
            return r2_score(y_test, y_pred)


    def save(self, filename="knn_player_price.pkl", path="models"):

        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        joblib.dump(self, full_path)
        print(f"Model saved at: {full_path}")

    # =========================
    # Load model
    # =========================
    @staticmethod
    def load(filepath="models/knn_player_price.pkl"):
        """
        Load model from .pkl file
        """
        model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return model
