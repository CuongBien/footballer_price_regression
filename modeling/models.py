import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class KNNPlayerPriceModel:
    """
    KNN model wrapper for football player price prediction
    Works with a pre-defined sklearn Pipeline
    """

    def __init__(self, pipeline):
        """
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Pipeline đã được xây dựng sẵn (preprocessing + KNNRegressor)
        """
        self.pipeline = pipeline
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Train the model

        Parameters
        ----------
        X_train : pandas.DataFrame or numpy.ndarray
        y_train : pandas.Series or numpy.ndarray
        """
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict player prices

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.pipeline.predict(X)

    def score(self, X_test, y_test, metric="rmse"):
        """
        Evaluate model performance

        Parameters
        ----------
        metric : str
            'rmse' | 'mse' | 'r2'
        """
        y_pred = self.predict(X_test)

        if metric == "rmse":
            return np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == "mse":
            return mean_squared_error(y_test, y_pred)
        elif metric == "r2":
            return r2_score(y_test, y_pred)
        else:
            raise ValueError("Unsupported metric. Use 'rmse', 'mse', or 'r2'.")
