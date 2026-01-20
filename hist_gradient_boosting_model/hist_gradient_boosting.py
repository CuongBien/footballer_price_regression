"""
=============================================================================
HISTGRADIENTBOOSTING MODEL - DỰ ĐOÁN GIÁ TRỊ CẦU THỦ
=============================================================================
Mô tả: Class wrapper cho HistGradientBoostingRegressor với 3 phương thức:
       - fit(): Huấn luyện model (bao gồm GridSearchCV tìm tham số tối ưu)
       - predict(): Dự đoán giá trị  
       - score(): Tính điểm R²
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from typing import Union, Dict, Optional
import time
import warnings

warnings.filterwarnings('ignore')


class HistGradientBoostingModel:
    """
    Custom wrapper class cho HistGradientBoostingRegressor
    
    3 phương thức chính:
    - fit(): Huấn luyện model với GridSearchCV tìm tham số tối ưu
    - predict(): Dự đoán giá trị
    - score(): Tính điểm R²
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 max_depth: int = None,
                 min_samples_leaf: int = 20,
                 l2_regularization: float = 0.0,
                 random_state: int = 42):
        """
        Khởi tạo HistGradientBoosting Model
        
        Parameters:
        -----------
        learning_rate : float
            Tốc độ học
        max_iter : int
            Số lượng iterations
        max_depth : int
            Độ sâu tối đa của mỗi cây
        min_samples_leaf : int
            Số samples tối thiểu trong leaf node
        l2_regularization : float
            L2 regularization
        random_state : int
            Random seed
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        
        self.model = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state
        )
        
        self.is_fitted = False
        self.best_params = None
        self.training_time = None
    

    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            param_grid: Optional[Dict] = None,
            cv: int = 5,
            scoring: str = 'neg_mean_absolute_error') -> 'HistGradientBoostingModel':
        """
        Huấn luyện model với GridSearchCV để tìm hyperparameters tối ưu
        
        Parameters:
        -----------
        X : pd.DataFrame hoặc np.ndarray
            Features (n_samples, n_features)
        y : pd.Series hoặc np.ndarray
            Target values (n_samples,)
        param_grid : Dict, optional
            Hyperparameters để search. Default:
            {
                'max_iter': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_leaf': [10, 20],
                'l2_regularization': [0, 0.1]
            }
        cv : int, default=5
            Số folds cross-validation
        scoring : str, default='neg_mean_absolute_error'
            Metric để đánh giá
            
        Returns:
        --------
        self : HistGradientBoostingModel
        """
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'max_iter': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_leaf': [10, 20],
                'l2_regularization': [0, 0.1]
            }
        
        # Tính tổng combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"\n Parameter Grid:")
        for param, values in param_grid.items():
            print(f"   - {param}: {values}")
        print(f"\n   Tổng combinations: {total_combinations}")
        print(f"   CV folds: {cv}")
        print(f"   Tổng lần training: {total_combinations * cv}")
        
        print(f"\n Đang tìm kiếm hyperparameters tối ưu...")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=HistGradientBoostingRegressor(random_state=self.random_state),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X, y)
        self.training_time = time.time() - start_time
        
        print(f"\n Training completed in {self.training_time:.2f} seconds")
        
        # Cập nhật model với best params
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        # Cập nhật hyperparameters
        self.learning_rate = self.best_params.get('learning_rate', self.learning_rate)
        self.max_iter = self.best_params.get('max_iter', self.max_iter)
        self.max_depth = self.best_params.get('max_depth', self.max_depth)
        self.min_samples_leaf = self.best_params.get('min_samples_leaf', self.min_samples_leaf)
        self.l2_regularization = self.best_params.get('l2_regularization', self.l2_regularization)
        
        # In kết quả
        print("\n BEST HYPERPARAMETERS:")
        print("-" * 50)
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        best_score = -grid_search.best_score_ if 'neg_' in scoring else grid_search.best_score_
        print(f"\n Best CV Score: {best_score:,.2f}")
        
        return self
    
 
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dự đoán giá trị cho dữ liệu mới
        
        Parameters:
        -----------
        X : pd.DataFrame hoặc np.ndarray
            Features (n_samples, n_features)
            
        Returns:
        --------
        predictions : np.ndarray
            Giá trị dự đoán
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        return self.model.predict(X)
    

    def score(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray]) -> float:
        """
        Tính điểm R² (coefficient of determination)
        
        Parameters:
        -----------
        X : Features
        y : True values
            
        Returns:
        --------
        score : float
            R² score (1 = hoàn hảo, 0 = chỉ predict mean)
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        return self.model.score(X, y)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("TEST HistGradientBoostingModel - 3 METHODS")
    print("=" * 70)
    
    # Tạo dữ liệu test
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test fit (đã bao gồm GridSearchCV)
    print("\n1️⃣ fit() - với GridSearchCV tự động")
    model = HistGradientBoostingModel(random_state=42)
    model.fit(X_train, y_train, 
              param_grid={'max_iter': [50, 100], 'learning_rate': [0.05, 0.1]},
              cv=3)
    
    # Test predict
    print("\n2️⃣ predict()")
    predictions = model.predict(X_test)
    print(f"   ✅ Predictions shape: {predictions.shape}")
    
    # Test score
    print("\n3️⃣ score()")
    r2 = model.score(X_test, y_test)
    print(f"   ✅ R² Score: {r2:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ ALL 3 METHODS WORK!")
    print("=" * 70)
