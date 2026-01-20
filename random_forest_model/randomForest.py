"""
Custom Random Forest Regressor
Sử dụng Custom Regression Tree đã implement
"""

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path để import custom RegressionTree
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from decisionTree.Regression.regressionTree import RegressionTree
from joblib import Parallel, delayed


class RandomForestRegressor:
    """
    Custom Random Forest Regressor từ Custom Regression Trees
    
    API giống sklearn: fit(X, y), predict(X), score(X, y)
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Số lượng cây trong forest
    criterion : str, default='mse'
        'mse' hoặc 'mae'
    max_depth : int, optional
        Độ sâu tối đa của mỗi cây
    min_samples_split : int, default=2
        Số samples tối thiểu để split
    min_samples_leaf : int, default=1
        Số samples tối thiểu ở mỗi lá
    max_features : int, float, str, default='sqrt'
        Số features tối đa xem xét mỗi split
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - int: số cụ thể
        - float: tỷ lệ %
        - None: tất cả features
    bootstrap : bool, default=True
        Có sử dụng bootstrap sampling không
    random_state : int, optional
        Random seed
    n_jobs : int, default=-1
        Số threads song song (-1 = tất cả cores)
    """
    
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                 bootstrap=True, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.trees = []
        self.n_features_ = None
        self.feature_names_ = None
        
    def _create_tree(self):
        """Tạo một cây mới với config"""
        return RegressionTree(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )
    
    def _bootstrap_sample(self, X, y, rng):
        """Tạo bootstrap sample"""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _fit_single_tree(self, X, y, seed):
        """Fit một cây đơn"""
        rng = np.random.RandomState(seed)
        
        # Bootstrap sampling
        if self.bootstrap:
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)
        else:
            X_sample, y_sample = X, y
        
        # Fit tree
        tree = self._create_tree()
        tree.fit(X_sample, y_sample)
        return tree
    
    def fit(self, X, y):
        """
        Huấn luyện Random Forest
        
        Parameters:
        -----------
        X : array-like hoặc pd.DataFrame
            Features (n_samples, n_features)
        y : array-like hoặc pd.Series
            Target values (n_samples,)
            
        Returns:
        --------
        self : RandomForestRegressor
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.n_features_ = X.shape[1]
        
        # Set random seeds for each tree
        if self.random_state is not None:
            np.random.seed(self.random_state)
        seeds = [np.random.randint(0, 10000) for _ in range(self.n_estimators)]
        
        print(f"\nTraining {self.n_estimators} Custom Regression Trees...")
        
        # Train trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._fit_single_tree)(X, y, seed) 
            for seed in seeds
        )
        
        print(f"✓ Trained {len(self.trees)} trees successfully!")
        
        return self
    
    def predict(self, X):
        """
        Dự đoán bằng cách average predictions từ tất cả các cây
        
        Parameters:
        -----------
        X : array-like hoặc pd.DataFrame
            Features (n_samples, n_features)
            
        Returns:
        --------
        predictions : np.ndarray
            Giá trị dự đoán (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def score(self, X, y):
        """
        Tính R² score
        
        Parameters:
        -----------
        X : array-like hoặc pd.DataFrame
            Features
        y : array-like hoặc pd.Series
            True values
            
        Returns:
        --------
        r2_score : float
            R² score (1 = hoàn hảo, 0 = chỉ predict mean)
        """
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64)
        
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    def get_params(self, deep=True):
        """Get parameters - for sklearn compatibility"""
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params):
        """Set parameters - for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("=" * 80)
    print("TEST Custom Random Forest Regressor")
    print("=" * 80)
    
    # Tạo dữ liệu test
    print("\nTạo dữ liệu test...")
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train model
    print("\nTraining Custom Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=50,
        criterion='mse',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Predict
    print("\nMaking predictions...")
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Train R² Score: {rf.score(X_train, y_train):.4f}")
    print(f"Test R² Score:  {rf.score(X_test, y_test):.4f}")
    print(f"Train RMSE:     {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
    print(f"Test RMSE:      {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
    print("=" * 80)
    
    # Compare with sklearn
    from sklearn.ensemble import RandomForestRegressor as SklearnRF
    print("\nComparing with Sklearn RandomForest...")
    sklearn_rf = SklearnRF(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    sklearn_rf.fit(X_train, y_train)
    print(f"Sklearn Test R²: {sklearn_rf.score(X_test, y_test):.4f}")
    print(f"Custom Test R²:  {rf.score(X_test, y_test):.4f}")
    
    print("\n✅ Custom Random Forest works!")
