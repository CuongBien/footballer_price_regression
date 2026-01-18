"""
Module: Imputation
Fill missing values with appropriate strategies
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class Imputer:
    """Class to handle missing value imputation"""
    
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent', knn_neighbors=5):
        """
        Parameters:
        -----------
        numeric_strategy : str
            Strategy for numeric columns: 'mean', 'median', 'most_frequent', 'knn'
        categorical_strategy : str
            Strategy for categorical columns: 'most_frequent', 'constant'
        knn_neighbors : int
            Number of neighbors for KNN imputation
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.knn_neighbors = knn_neighbors
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_cols = []
        self.categorical_cols = []
        
    def fit(self, X_train: pd.DataFrame):
        """Fit imputers on training data"""
        self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.numeric_cols:
            if self.numeric_strategy == 'knn':
                self.numeric_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            else:
                self.numeric_imputer = SimpleImputer(strategy=self.numeric_strategy)
            self.numeric_imputer.fit(X_train[self.numeric_cols])
        
        if self.categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(X_train[self.categorical_cols])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted imputers"""
        X_imputed = X.copy()
        
        if self.numeric_cols and self.numeric_imputer:
            X_imputed[self.numeric_cols] = self.numeric_imputer.transform(X[self.numeric_cols])
        
        if self.categorical_cols and self.categorical_imputer:
            X_imputed[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
        
        return X_imputed
    
    def fit_transform(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None):
        """Fit imputers on train and transform all sets"""
        total_missing = X_train.isnull().sum().sum()
        
        if total_missing == 0:
            print("No missing values found.")
            if X_val is not None and X_test is not None:
                return X_train, X_val, X_test
            return X_train
        
        print(f"Total missing values in training set: {total_missing}")
        
        self.fit(X_train)
        X_train_imputed = self.transform(X_train)
        
        if X_val is not None and X_test is not None:
            X_val_imputed = self.transform(X_val)
            X_test_imputed = self.transform(X_test)
            print(f"✓ Imputation completed")
            return X_train_imputed, X_val_imputed, X_test_imputed
        
        print(f"✓ Imputation completed")
        return X_train_imputed
    
    def get_feature_names(self):
        """Get feature names after imputation"""
        return self.numeric_cols + self.categorical_cols

