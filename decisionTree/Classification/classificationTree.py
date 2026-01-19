"""
Custom Classification Tree - Optimized Version
Sử dụng binary threshold splits cho continuous data
Tối ưu hóa với numpy vectorization
"""

import numpy as np
import pandas as pd


class Node:
    """Đại diện cho một nút trong cây quyết định"""
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    Cây quyết định phân loại tối ưu với binary threshold splits
    
    API giống sklearn: fit(X, y), predict(X), score(X, y)
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None):
        self.criterion = criterion  # 'gini' hoặc 'entropy'/'information_gain'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_features_ = None
        self.feature_names_ = None
        self.classes_ = None
    
    def _gini(self, y):
        """Tính Gini impurity"""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _entropy(self, y):
        """Tính Entropy"""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def _compute_impurity(self, y):
        """Tính impurity theo criterion"""
        if self.criterion == 'gini':
            return self._gini(y)
        else:  # entropy / information_gain
            return self._entropy(y)
    
    def _compute_impurity_reduction(self, y, left_mask):
        """Tính information gain / gini reduction"""
        right_mask = ~left_mask
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = len(y)
        
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return -np.inf
        
        parent_impurity = self._compute_impurity(y)
        left_impurity = self._compute_impurity(y[left_mask])
        right_impurity = self._compute_impurity(y[right_mask])
        
        weighted_child_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        return parent_impurity - weighted_child_impurity
    
    def _get_n_features_to_sample(self, n_features):
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
        """Tìm split tốt nhất"""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
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
            
            if len(unique_vals) > 50:
                percentiles = np.percentile(unique_vals, np.linspace(0, 100, 51))
                thresholds = np.unique(percentiles)[:-1]
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                gain = self._compute_impurity_reduction(y, left_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _most_common_label(self, y):
        """Trả về label phổ biến nhất"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def _build_tree(self, X, y, depth=0):
        """Xây dựng cây đệ quy"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf or \
           n_classes == 1:
            return Node(value=self._most_common_label(y))
        
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return Node(value=self._most_common_label(y))
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return Node(value=self._most_common_label(y))
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def fit(self, X, y):
        """Huấn luyện cây quyết định"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_single(self.root, x) for x in X])
    
    def score(self, X, y):
        """Tính accuracy"""
        if isinstance(y, pd.Series):
            y = y.values
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Aliases for backward compatibility
DecisionTree = DecisionTreeClassifier
CustomDecisionTreeClassifier = DecisionTreeClassifier
