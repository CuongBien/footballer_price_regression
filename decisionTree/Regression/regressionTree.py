import numpy as np
import pandas as pd
from typing import Union, Dict, Any

from Metrics.MSE import compute_MSE_Reduction
from Metrics.MAE import compute_MAE_Reduction


class Node:
    """Đại diện cho một nút trong cây hồi quy"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.children = {}
        
    def is_leaf(self):
        return self.value is not None


class RegressionTree:
    """Cây hồi quy hỗ trợ MSE và MAE criterion"""
    
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        self.target_name = None
        
    def find_best_split(self, data, features, target):
        """Tìm feature tốt nhất để split"""
        best_reduction = -1
        best_feature = None
        
        for feature in features:
            if self.criterion == 'mse':
                reduction = compute_MSE_Reduction(data, feature, target)
            else:
                reduction = compute_MAE_Reduction(data, feature, target)
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = feature
                
        return best_feature, best_reduction
    
    def leaf_value(self, target_column):
        if self.criterion == 'mse':
            return np.mean(target_column)
        else:
            return np.median(target_column)
    
    def build_tree(self, data, features, target, depth=0):
        """Xây dựng cây đệ quy"""
        target_column = data[target]
        
        # Điều kiện dừng
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=self.leaf_value(target_column))
        
        if len(data) < self.min_samples_split:
            return Node(value=self.leaf_value(target_column))
        
        if len(np.unique(target_column)) == 1:
            return Node(value=target_column.iloc[0])
        
        # Tìm feature tốt nhất
        best_feature, best_reduction = self.find_best_split(data, features, target)
        
        if best_reduction <= 0:
            return Node(value=self.leaf_value(target_column))
        
        # Tạo node và các nhánh con
        node = Node(feature=best_feature)
        unique_values = data[best_feature].unique()
        remaining_features = [f for f in features if f != best_feature]
        
        for value in unique_values:
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                node.children[value] = Node(value=self.leaf_value(target_column))
            else:
                node.children[value] = self.build_tree(subset, remaining_features, target, depth + 1)
        
        return node
    
    def fit(self, data, target_name):
        """Huấn luyện cây hồi quy"""
        self.target_name = target_name
        self.feature_names = [col for col in data.columns if col != target_name]
        self.root = self.build_tree(data, self.feature_names, target_name)
        return self
    
    def predict_single(self, node, sample):
        """Dự đoán cho một mẫu"""
        if node.is_leaf():
            return node.value
        
        feature_value = sample[node.feature]
        
        if feature_value in node.children:
            return self.predict_single(node.children[feature_value], sample)
        else:
            if len(node.children) > 0:
                first_child = list(node.children.values())[0]
                return self.predict_single(first_child, sample)
            return node.value if node.is_leaf() else 0
    
    def predict(self, data):
        """Dự đoán cho nhiều mẫu"""
        predictions = []
        for idx in range(len(data)):
            sample = data.iloc[idx]
            prediction = self.predict_single(self.root, sample)
            predictions.append(prediction)
        return predictions
    
    def print_tree(self, node=None, depth=0, prefix="Root"):
        """In cấu trúc cây"""
        if node is None:
            node = self.root
            
        indent = "  " * depth
        
        if node.is_leaf():
            print(f"{indent}{prefix} -> Dự đoán: {node.value:.2f}")
        else:
            print(f"{indent}{prefix} -> Split theo: {node.feature}")
            for value, child in node.children.items():
                self.print_tree(child, depth + 1, f"{node.feature} = {value}")
    
    def score(self, data, target_name):
        """Tính R² score"""
        predictions = self.predict(data)
        actual = data[target_name].values
        
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def mse_score(self, data, target_name):
        """Tính Mean Squared Error"""
        predictions = self.predict(data)
        actual = data[target_name].values
        return np.mean((actual - predictions) ** 2)
    
    def mae_score(self, data, target_name):
        """Tính Mean Absolute Error"""
        predictions = self.predict(data)
        actual = data[target_name].values
        return np.mean(np.abs(actual - predictions))
