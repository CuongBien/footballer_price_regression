import numpy as np
import pandas as pd
from typing import Union, Dict, Any

from Metrics.MSE import compute_MSE_Reduction
from Metrics.MAE import compute_MAE_Reduction


class Node:
    """Đại diện cho một nút trong cây hồi quy"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Thuộc tính để split
        self.threshold = threshold  # Giá trị để split (cho continuous) hoặc None
        self.left = left           # Nhánh trái
        self.right = right         # Nhánh phải
        self.value = value         # Giá trị dự đoán nếu là lá
        self.children = {}         # Dictionary cho categorical splits
        
    def is_leaf(self):
        return self.value is not None


class RegressionTree:
    """Cây hồi quy hỗ trợ cả MSE và MAE"""
    
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        self.target_name = None
        
    def find_best_split(self, data, features, target):
        """Tìm feature tốt nhất để split - sử dụng các module đã import"""
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
        
        # Điều kiện dừng: nếu không còn features hoặc đạt max_depth
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=self.leaf_value(target_column))
        
        # Nếu ít hơn min_samples_split, trả về lá
        if len(data) < self.min_samples_split:
            return Node(value=self.leaf_value(target_column))
        
        # Nếu tất cả giá trị target giống nhau, trả về lá
        if len(np.unique(target_column)) == 1:
            return Node(value=target_column.iloc[0])
        
        # Tìm feature tốt nhất để split
        best_feature, best_reduction = self.find_best_split(data, features, target)
        
        # Nếu không có reduction (hoặc reduction = 0), trả về lá
        if best_reduction <= 0:
            return Node(value=self.leaf_value(target_column))
        
        # Tạo node với feature tốt nhất
        node = Node(feature=best_feature)
        
        # Lấy các giá trị unique của feature
        unique_values = data[best_feature].unique()
        
        # Tạo các nhánh con cho mỗi giá trị
        remaining_features = [f for f in features if f != best_feature]
        
        for value in unique_values:
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                # Nếu subset rỗng, tạo lá với giá trị trung bình
                node.children[value] = Node(value=self.leaf_value(target_column))
            else:
                # Đệ quy xây dựng cây con
                node.children[value] = self.build_tree(subset, remaining_features, target, depth + 1)
        
        return node
    
    def fit(self, data, target_name):
        """
        Huấn luyện cây hồi quy
        """
        self.target_name = target_name
        self.feature_names = [col for col in data.columns if col != target_name]
        self.root = self.build_tree(data, self.feature_names, target_name)
        return self
    
    def predict_single(self, node, sample):
        """Dự đoán cho một mẫu"""
        if node.is_leaf():
            return node.value
        
        feature_value = sample[node.feature]
        
        # Nếu giá trị này tồn tại trong children
        if feature_value in node.children:
            return self.predict_single(node.children[feature_value], sample)
        else:
            # Nếu không tìm thấy giá trị, trả về giá trị của nhánh đầu tiên (fallback)
            if len(node.children) > 0:
                first_child = list(node.children.values())[0]
                return self.predict_single(first_child, sample)
            return node.value if node.is_leaf() else 0
    
    def predict(self, data):
        """
        Dự đoán cho nhiều mẫu
        """
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
        """
        Tính R² score (coefficient of determination)
        """
        predictions = self.predict(data)
        actual = data[target_name].values
        
        # Tính R² score
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
        mse = np.mean((actual - predictions) ** 2)
        return mse
    
    def mae_score(self, data, target_name):
        """Tính Mean Absolute Error"""
        predictions = self.predict(data)
        actual = data[target_name].values
        mae = np.mean(np.abs(actual - predictions))
        return mae


# === DEMO ===
if __name__ == "__main__":
    # Tạo dữ liệu mẫu cho regression
    data = {
        'Size': ['Small', 'Small', 'Medium', 'Large', 'Large', 'Large', 'Medium', 'Small', 'Medium', 'Large'],
        'Location': ['City', 'Suburb', 'City', 'City', 'Suburb', 'Suburb', 'Suburb', 'City', 'City', 'Suburb'],
        'Rooms': [1, 2, 2, 3, 3, 4, 2, 1, 2, 3],
        'Price': [150, 180, 250, 400, 350, 380, 220, 170, 240, 390]
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("DEMO CÂY HỒI QUY (REGRESSION TREE)")
    print("=" * 60)
    print("\nDữ liệu:")
    print(df)
    
    # Test với MSE
    print("\n" + "=" * 60)
    print("1. CÂY HỒI QUY VỚI MSE")
    print("=" * 60)
    tree_mse = RegressionTree(criterion='mse', max_depth=3)
    tree_mse.fit(df, 'Price')
    print("\nCấu trúc cây:")
    tree_mse.print_tree()
    
    r2 = tree_mse.score(df, 'Price')
    mse = tree_mse.mse_score(df, 'Price')
    print(f"\nR² score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    
    # Test với MAE
    print("\n" + "=" * 60)
    print("2. CÂY HỒI QUY VỚI MAE")
    print("=" * 60)
    tree_mae = RegressionTree(criterion='mae', max_depth=3)
    tree_mae.fit(df, 'Price')
    print("\nCấu trúc cây:")
    tree_mae.print_tree()
    
    r2 = tree_mae.score(df, 'Price')
    mae = tree_mae.mae_score(df, 'Price')
    print(f"\nR² score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    
    # Test dự đoán
    print("\n" + "=" * 60)
    print("3. DỰ ĐOÁN MẪU MỚI")
    print("=" * 60)
    test_data = pd.DataFrame({
        'Size': ['Medium', 'Large', 'Small'],
        'Location': ['City', 'Suburb', 'City'],
        'Rooms': [2, 3, 1]
    })
    print("\nDữ liệu test:")
    print(test_data)
    
    predictions_mse = tree_mse.predict(test_data)
    predictions_mae = tree_mae.predict(test_data)
    
    print("\nKết quả dự đoán giá:")
    print(f"MSE criterion: {[f'{p:.2f}' for p in predictions_mse]}")
    print(f"MAE criterion: {[f'{p:.2f}' for p in predictions_mae]}")
    print("=" * 60)
