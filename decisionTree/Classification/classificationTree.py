import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from Metrics.giniImpurity import compute_Gini_Gain
from Metrics.informationGain import compute_information_gain

class Node:
    """Đại diện cho một nút trong cây quyết định"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Thuộc tính để split
        self.threshold = threshold  # Giá trị để split (cho continuous) hoặc None
        self.left = left           # Nhánh trái
        self.right = right         # Nhánh phải
        self.value = value         # Giá trị dự đoán nếu là lá
        self.children = {}         # Dictionary cho categorical splits
        
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """Cây quyết định hỗ trợ cả Information Gain và Gini Impurity"""
    def __init__(self, criterion='information_gain', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        self.target_name = None
    
    def find_best_split(self, data, features, target):
        """Tìm feature tốt nhất để split"""
        best_gain = -1
        best_feature = None
        
        for feature in features:
            if self.criterion == 'information_gain':
                gain = compute_information_gain(data, feature, target)
            else:
                gain = compute_Gini_Gain(data, feature, target)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                
        return best_feature, best_gain
    
    def most_common_label(self, target_column):
        """Trả về nhãn phổ biến nhất"""
        return target_column.mode()[0]
    
    def build_tree(self, data, features, target, depth=0):
        """Xây dựng cây đệ quy"""
        # Điều kiện dừng
        target_column = data[target]
        
        # Nếu tất cả nhãn giống nhau, trả về lá
        if len(np.unique(target_column)) == 1:
            return Node(value=target_column.iloc[0])
        
        # Nếu không còn features hoặc đạt max_depth, trả về lá với nhãn phổ biến nhất
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=self.most_common_label(target_column))
        
        # Nếu ít hơn min_samples_split, trả về lá
        if len(data) < self.min_samples_split:
            return Node(value=self.most_common_label(target_column))
        
        # Tìm feature tốt nhất để split
        best_feature, best_gain = self.find_best_split(data, features, target)
        
        # Nếu không có gain, trả về lá
        if best_gain == 0:
            return Node(value=self.most_common_label(target_column))
        
        # Tạo node với feature tốt nhất
        node = Node(feature=best_feature)
        
        # Lấy các giá trị unique của feature
        unique_values = data[best_feature].unique()
        
        # Tạo các nhánh con cho mỗi giá trị
        remaining_features = [f for f in features if f != best_feature]
        
        for value in unique_values:
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                # Nếu subset rỗng, tạo lá với nhãn phổ biến nhất
                node.children[value] = Node(value=self.most_common_label(target_column))
            else:
                # Đệ quy xây dựng cây con
                node.children[value] = self.build_tree(subset, remaining_features, target, depth + 1)
        
        return node
    
    def fit(self, data, target_name):
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
            return None
    
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
            print(f"{indent}{prefix} -> Dự đoán: {node.value}")
        else:
            print(f"{indent}{prefix} -> Split theo: {node.feature}")
            for value, child in node.children.items():
                self.print_tree(child, depth + 1, f"{node.feature} = {value}")
    
    def score(self, data, target_name):
        """Tính accuracy"""
        predictions = self.predict(data)
        actual = data[target_name].values
        correct = sum([1 for pred, act in zip(predictions, actual) if pred == act])
        return correct / len(actual)


# === DEMO ===
if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    data = {
        'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
        'Temp':    ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
        'Play':    ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("DEMO CÂY QUYẾT ĐỊNH")
    print("=" * 60)
    print("\nDữ liệu:")
    print(df)
    
    # Test với Information Gain
    print("\n" + "=" * 60)
    print("1. CÂY QUYẾT ĐỊNH VỚI INFORMATION GAIN")
    print("=" * 60)
    tree_ig = DecisionTree(criterion='information_gain', max_depth=3)
    tree_ig.fit(df, 'Play')
    print("\nCấu trúc cây:")
    tree_ig.print_tree()
    
    accuracy = tree_ig.score(df, 'Play')
    print(f"\nĐộ chính xác trên tập huấn luyện: {accuracy * 100:.2f}%")
    
    # Test với Gini
    print("\n" + "=" * 60)
    print("2. CÂY QUYẾT ĐỊNH VỚI GINI IMPURITY")
    print("=" * 60)
    tree_gini = DecisionTree(criterion='gini', max_depth=3)
    tree_gini.fit(df, 'Play')
    print("\nCấu trúc cây:")
    tree_gini.print_tree()
    
    accuracy = tree_gini.score(df, 'Play')
    print(f"\nĐộ chính xác trên tập huấn luyện: {accuracy * 100:.2f}%")
    
    # Test dự đoán
    print("\n" + "=" * 60)
    print("3. DỰ ĐOÁN MẪU MỚI")
    print("=" * 60)
    test_data = pd.DataFrame({
        'Weather': ['Sunny', 'Overcast', 'Rainy'],
        'Temp': ['Cool', 'Hot', 'Mild']
    })
    print("\nDữ liệu test:")
    print(test_data)
    
    predictions_ig = tree_ig.predict(test_data)
    predictions_gini = tree_gini.predict(test_data)
    
    print("\nKết quả dự đoán:")
    print(f"Information Gain: {predictions_ig}")
    print(f"Gini Impurity:    {predictions_gini}")
    print("=" * 60)
