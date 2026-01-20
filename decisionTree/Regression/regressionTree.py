"""
Custom Regression Tree - Optimized Version
Sử dụng binary threshold splits cho continuous data (giống sklearn)
Tối ưu hóa với numpy vectorization
"""

import numpy as np
import pandas as pd


class Node:
    """Đại diện cho một nút trong cây hồi quy"""
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index của feature để split
        self.threshold = threshold  # Ngưỡng để split (<=)
        self.left = left           # Nhánh trái (<=)
        self.right = right         # Nhánh phải (>)
        self.value = value         # Giá trị dự đoán nếu là lá
    
    def is_leaf(self):
        return self.value is not None


class RegressionTree:
    """
    Cây hồi quy tối ưu với binary threshold splits
    
    API giống sklearn: fit(X, y), predict(X), score(X, y)
    
    Parameters:
    -----------
    criterion : str
        'mse' hoặc 'mae'
    max_depth : int, optional
        Độ sâu tối đa của cây
    min_samples_split : int
        Số samples tối thiểu để split
    min_samples_leaf : int
        Số samples tối thiểu ở mỗi lá
    max_features : int, float, str, optional
        Số features tối đa xem xét mỗi split
    """
    
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_features_ = None
        self.feature_names_ = None
    
    def _compute_leaf_value(self, y):
        """Tính giá trị dự đoán cho leaf node"""
        if self.criterion == 'mse':
            return np.mean(y)
        else:  # mae
            return np.median(y)
    
    def _compute_impurity(self, y):
        """Tính impurity (MSE hoặc MAE)"""
        if len(y) == 0:
            return 0.0
        if self.criterion == 'mse':
            return np.var(y) * len(y)
        else:  # mae
            return np.sum(np.abs(y - np.median(y)))
    
    def _compute_impurity_reduction(self, y, left_mask):
        """Tính reduction khi split"""
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return -np.inf
        
        parent_impurity = self._compute_impurity(y)
        left_impurity = self._compute_impurity(y[left_mask])
        right_impurity = self._compute_impurity(y[right_mask])
        
        return parent_impurity - left_impurity - right_impurity
    
    def _get_n_features_to_sample(self, n_features):
        """Xác định số features để sample"""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        return n_features
    
    def _find_best_split(self, X, y):
        """Tìm split tốt nhất - VECTORIZED & OPTIMIZED"""
        n_samples, n_features = X.shape
        
        best_reduction = -np.inf
        best_feature = None
        best_threshold = None
        
        # Sample features nếu cần
        n_features_to_sample = self._get_n_features_to_sample(n_features)
        if n_features_to_sample < n_features:
            feature_indices = np.random.choice(n_features, n_features_to_sample, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]
            unique_vals = np.unique(feature_values)
            
            if len(unique_vals) <= 1:
                continue
            
            # Sample thresholds nếu quá nhiều (tăng tốc)
            if len(unique_vals) > 50:
                percentiles = np.percentile(unique_vals, np.linspace(0, 100, 51))
                thresholds = np.unique(percentiles)[:-1]
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                reduction = self._compute_impurity_reduction(y, left_mask)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feat_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X, y, depth=0):
        """Xây dựng cây đệ quy"""
        n_samples = len(y)
        
        # Stopping conditions - FIXED: Không check 2*min_samples_leaf
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           np.all(y == y[0]):
            return Node(value=self._compute_leaf_value(y))
        
        # Find best split
        best_feature, best_threshold, best_reduction = self._find_best_split(X, y)
        
        # Thêm điều kiện min_impurity_decrease giống sklearn
        if best_feature is None or best_reduction <= self.min_impurity_decrease:
            return Node(value=self._compute_leaf_value(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check min_samples_leaf sau khi split (đã check trong _compute_impurity_reduction)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return Node(value=self._compute_leaf_value(y))
        
        # Build children recursively
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def fit(self, X, y):
        """Huấn luyện cây hồi quy"""
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
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, node, x):
        """Dự đoán cho một sample"""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)
    
    def predict(self, X):
        """Dự đoán cho nhiều samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_single(self.root, x) for x in X])
    
    def score(self, X, y):
        """Tính R² score"""
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=np.float64)
        
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    def get_depth(self, node=None):
        """Lấy độ sâu của cây"""
        if node is None:
            node = self.root
        if node.is_leaf():
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))


# Alias for backward compatibility  
CustomRegressionTree = RegressionTree
