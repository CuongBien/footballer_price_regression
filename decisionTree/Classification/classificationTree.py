import numpy as np
import pandas as pd
from typing import Union, Dict, Any


def compute_entropy(target_column):
    """Tính entropy của một cột target"""
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = -np.sum([(count/np.sum(counts)) * np.log2(count/np.sum(counts)) 
                       for count in counts])
    return entropy


def compute_information_gain(data, split_feature, target_name):
    """Tính Information Gain khi split theo feature"""
    total_entropy = compute_entropy(data[target_name])
    
    vals, counts = np.unique(data[split_feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * 
                               compute_entropy(data.where(data[split_feature]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain


def compute_gini_impurity(target_column):
    """Tính Gini Impurity"""
    elements, counts = np.unique(target_column, return_counts=True)
    probabilities = counts / np.sum(counts)
    gini = 1 - np.sum(probabilities ** 2)
    return gini


def compute_Gini_Gain(data, split_feature, target_name):
    """Tính Gini Gain khi split theo feature"""
    total_gini = compute_gini_impurity(data[target_name])
    
    vals, counts = np.unique(data[split_feature], return_counts=True)
    weighted_gini = np.sum([(counts[i]/np.sum(counts)) * 
                            compute_gini_impurity(data.where(data[split_feature]==vals[i]).dropna()[target_name])
                            for i in range(len(vals))])
    
    gini_gain = total_gini - weighted_gini
    return gini_gain


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
    # Đọc dữ liệu FIFA
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '../../data/sofifa_players.csv')
    df = pd.read_csv(csv_path)
    
    # Chọn các features và target
    selected_cols = ['Age', 'Overall', 'Potential', 'Wage_Numeric', 'Value_Numeric']
    df = df[selected_cols].dropna()
    
    # Loại bỏ các hàng có Value_Numeric = 0
    df = df[df['Value_Numeric'] > 0]
    
    # Discretize numeric features thành categorical bins
    def discretize_column(col, bins, labels):
        return pd.cut(col, bins=bins, labels=labels, include_lowest=True)
    
    df_binned = df.copy()
    df_binned['Age_Cat'] = discretize_column(df['Age'], bins=[0, 22, 28, 35, 50], labels=['Young', 'Prime', 'Experienced', 'Veteran'])
    df_binned['Potential_Cat'] = discretize_column(df['Potential'], bins=[0, 75, 85, 90, 100], labels=['Low', 'Medium', 'High', 'Elite'])
    df_binned['Wage_Cat'] = discretize_column(df['Wage_Numeric'], bins=[-1, 30000, 80000, 150000, 1000000], labels=['Low', 'Medium', 'High', 'Elite'])
    
    # Target: Phân loại cầu thủ theo mức đánh giá Overall (hợp lý hơn Preferred_Foot)
    df_binned['Overall_Category'] = discretize_column(df['Overall'], bins=[0, 70, 80, 85, 100], labels=['Low', 'Medium', 'High', 'Elite'])
    
    # Balanced sampling: 100 samples per Overall_Category
    samples_per_class = 100
    balanced_dfs = []
    for category in ['Low', 'Medium', 'High', 'Elite']:
        category_df = df_binned[df_binned['Overall_Category'] == category]
        n_samples = min(samples_per_class, len(category_df))
        if n_samples > 0:
            sampled = category_df.sample(n=n_samples, random_state=42)
            balanced_dfs.append(sampled)
            print(f"Sampled {n_samples} from Overall_Category={category}")
    
    df_binned = pd.concat(balanced_dfs, ignore_index=True)
    
    target_column = 'Overall_Category'
    features = ['Age_Cat', 'Potential_Cat', 'Wage_Cat']
    
    train_df = df_binned[features + [target_column]].copy()
    
    print("=" * 60)
    print("DEMO CÂY QUYẾT ĐỊNH (CLASSIFICATION TREE)")
    print("Dự đoán mức đánh giá cầu thủ (Overall_Category)")
    print("=" * 60)
    print(f"\nSố lượng mẫu: {len(train_df)}")
    print(f"Features: {features}")
    print(f"Target: {target_column}")
    print(f"\nPhân phối target:")
    print(train_df[target_column].value_counts().sort_index())
    print(f"\nBaseline accuracy (đoán class phổ biến nhất): {train_df[target_column].value_counts().max() / len(train_df) * 100:.1f}%")
    print("\nDữ liệu mẫu:")
    print(train_df.head(10).to_string())
    
    # Test với Information Gain
    print("\n" + "=" * 60)
    print("1. CÂY QUYẾT ĐỊNH VỚI INFORMATION GAIN")
    print("=" * 60)
    tree_ig = DecisionTree(criterion='information_gain', max_depth=4)
    tree_ig.fit(train_df, target_column)
    print("\nCấu trúc cây:")
    tree_ig.print_tree()
    
    accuracy_ig = tree_ig.score(train_df, target_column)
    print(f"\n📊 Kết quả Classification với Information Gain:")
    print(f"  Accuracy: {accuracy_ig * 100:.2f}%")
    
    # Test với Gini
    print("\n" + "=" * 60)
    print("2. CÂY QUYẾT ĐỊNH VỚI GINI IMPURITY")
    print("=" * 60)
    tree_gini = DecisionTree(criterion='gini', max_depth=4)
    tree_gini.fit(train_df, target_column)
    print("\nCấu trúc cây:")
    tree_gini.print_tree()
    
    accuracy_gini = tree_gini.score(train_df, target_column)
    print(f"\n📊 Kết quả Classification với Gini Impurity:")
    print(f"  Accuracy: {accuracy_gini * 100:.2f}%")
    
    # Test dự đoán
    print("\n" + "=" * 60)
    print("3. DỰ ĐOÁN MẪU MỚI")
    print("=" * 60)
    test_data = pd.DataFrame({
        'Age_Cat': ['Young', 'Prime', 'Experienced', 'Veteran'],
        'Potential_Cat': ['Elite', 'High', 'Medium', 'Low'],
        'Wage_Cat': ['Elite', 'High', 'Medium', 'Low']
    })
    print("\nDữ liệu test (4 cầu thủ mẫu):")
    print(test_data.to_string())
    
    predictions_ig = tree_ig.predict(test_data)
    predictions_gini = tree_gini.predict(test_data)
    
    print("\nKết quả dự đoán mức Overall:")
    print(f"Information Gain: {predictions_ig}")
    print(f"Gini Impurity:    {predictions_gini}")
    
    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    print(f"Information Gain Accuracy: {accuracy_ig * 100:.2f}%")
    print(f"Gini Impurity Accuracy:    {accuracy_gini * 100:.2f}%")
    print("=" * 60)


# Alias for sklearn-like naming
DecisionTreeClassifier = DecisionTree
