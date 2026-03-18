import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


class KNNModel:
    """
    K-Nearest Neighbors Model with GridSearchCV for hyperparameter tuning
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.is_fitted = False
        self.best_params = None
        self.grid_search_results = None

    def fit(self, X_train, y_train):
        """Fit the KNN model"""
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized")
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using the KNN model"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.pipeline.predict(X)
    
    def score(self, X_test, y_test, metric="rmse"):
        """Calculate score using specified metric"""
        y_pred = self.predict(X_test)

        if metric == "rmse":
            return np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == "mse":
            return mean_squared_error(y_test, y_pred)
        elif metric == "r2":
            return r2_score(y_test, y_pred)
        elif metric == "mae":
            return mean_absolute_error(y_test, y_pred)
    
    def hyperparameter_tuning(self, X_train, y_train, X_val=None, y_val=None, cv=5):
        """
        Perform GridSearchCV to find optimal hyperparameters for KNN
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_val : array-like, optional
            Validation features (for testing after CV)
        y_val : array-like, optional
            Validation target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Best parameters found
        """
        print("\n" + "="*70)
        print("KNN HYPERPARAMETER TUNING - GridSearchCV")
        print("="*70)
        
        # Define parameter grid for KNN
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        print("\nParameter Grid:")
        print("-" * 70)
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"\nTotal parameter combinations: {total_combinations}")
        print(f"Cross-validation folds: {cv}")
        print(f"Total model fits: {total_combinations * cv}")
        print("-" * 70)
        
        # Create base KNN model
        knn = KNeighborsRegressor()
        
        # Perform GridSearchCV
        print("\nSearching for optimal parameters...")
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)
        
        # Display results
        print("\n" + "="*70)
        print("BEST PARAMETERS FOUND")
        print("="*70)
        for param, value in self.best_params.items():
            print(f"  ✓ {param}: {value}")
        
        print(f"\n  Best CV Score (R² mean): {grid_search.best_score_:.6f}")
        
        # Test on validation set if provided
        if X_val is not None and y_val is not None:
            best_model = grid_search.best_estimator_
            val_r2 = best_model.score(X_val, y_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, best_model.predict(X_val)))
            val_mae = mean_absolute_error(y_val, best_model.predict(X_val))
            
            print("\n" + "="*70)
            print("VALIDATION SET PERFORMANCE")
            print("="*70)
            print(f"  - R² Score: {val_r2:.6f}")
            print(f"  - RMSE: {val_rmse:,.2f}")
            print(f"  - MAE: {val_mae:,.2f}")
        
        # Update the pipeline with best model
        self.pipeline = grid_search.best_estimator_
        self.is_fitted = True
        
        return self.best_params
    
    def get_top_parameters(self, top_n=10):
        """
        Get top N parameter combinations based on CV score
        
        Parameters:
        -----------
        top_n : int
            Number of top results to return
            
        Returns:
        --------
        pd.DataFrame : Top N parameter combinations with their scores
        """
        if self.grid_search_results is None:
            raise ValueError("GridSearchCV has not been performed yet")
        
        # Sort by mean test score
        top_results = self.grid_search_results.nlargest(top_n, 'mean_test_score')[
            ['param_n_neighbors', 'param_weights', 'param_metric', 'param_algorithm',
             'mean_test_score', 'std_test_score', 'rank_test_score']
        ]
        
        return top_results.reset_index(drop=True)
    
    def print_cv_summary(self):
        """Print summary of cross-validation results"""
        if self.grid_search_results is None:
            print("GridSearchCV has not been performed yet")
            return
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION SUMMARY")
        print("="*70)
        
        results = self.grid_search_results
        print(f"\nTotal configurations tested: {len(results)}")
        print(f"Best CV Score (R²): {results['mean_test_score'].max():.6f}")
        print(f"Worst CV Score (R²): {results['mean_test_score'].min():.6f}")
        print(f"Mean CV Score (R²): {results['mean_test_score'].mean():.6f}")
        print(f"Std Dev CV Score: {results['mean_test_score'].std():.6f}")
        
        print("\nTop 5 Parameter Combinations:")
        print("-" * 70)
        top_5 = self.get_top_parameters(5)
        for idx, row in top_5.iterrows():
            print(f"\n#{idx+1} - R² Score: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
            print(f"    n_neighbors: {row['param_n_neighbors']}")
            print(f"    weights: {row['param_weights']}")
            print(f"    metric: {row['param_metric']}")
            print(f"    algorithm: {row['param_algorithm']}")

    def save(self, filename="knn_player_price.pkl", path="models"):
        """Save model to disk"""
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        joblib.dump(self, full_path)
        print(f"✓ Model saved at: {full_path}")
        
        # Also save best parameters if available
        if self.best_params is not None:
            params_path = os.path.join(path, "knn_best_params.pkl")
            joblib.dump(self.best_params, params_path)
            print(f"✓ Best parameters saved at: {params_path}")

    @staticmethod
    def load(filepath="models/knn_player_price.pkl"):
        """Load model from .pkl file"""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from: {filepath}")
        return model
