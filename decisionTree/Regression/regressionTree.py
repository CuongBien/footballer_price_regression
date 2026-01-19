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
    # Đọc dữ liệu từ file CSV
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '../../data/sofifa_players.csv')
    df = pd.read_csv(csv_path)
    
    # Chọn các features và target
    selected_cols = ['Age', 'Overall', 'Potential', 'Wage_Numeric', 'Value_Numeric']
    df = df[selected_cols].dropna()
    
    # Loại bỏ các hàng có Value_Numeric = 0
    df = df[df['Value_Numeric'] > 0]
    
    # Chuyển Value sang triệu € để MSE dễ đọc hơn
    df['Value_Million'] = df['Value_Numeric'] / 1_000_000
    
    # Discretize numeric features
    def discretize_column(col, bins, labels):
        return pd.cut(col, bins=bins, labels=labels, include_lowest=True)
    
    df_binned = df.copy()
    df_binned['Age'] = discretize_column(df['Age'], bins=[0, 22, 28, 35, 50], labels=['Young', 'Prime', 'Experienced', 'Veteran'])
    df_binned['Overall'] = discretize_column(df['Overall'], bins=[0, 70, 80, 85, 100], labels=['Low', 'Medium', 'High', 'Elite'])
    df_binned['Potential'] = discretize_column(df['Potential'], bins=[0, 75, 85, 90, 100], labels=['Low', 'Medium', 'High', 'Elite'])
    df_binned['Wage_Numeric'] = discretize_column(df['Wage_Numeric'], bins=[-1, 30000, 80000, 150000, 1000000], labels=['Low', 'Medium', 'High', 'Elite'])
    
    # Balanced sampling: 100 samples per Overall category
    samples_per_class = 100
    balanced_dfs = []
    for category in ['Low', 'Medium', 'High', 'Elite']:
        category_df = df_binned[df_binned['Overall'] == category]
        n_samples = min(samples_per_class, len(category_df))
        if n_samples > 0:
            sampled = category_df.sample(n=n_samples, random_state=42)
            balanced_dfs.append(sampled)
            print(f"Sampled {n_samples} from Overall={category}")
    
    df_binned = pd.concat(balanced_dfs, ignore_index=True)
    
    target_column = 'Value_Million'  # Dùng triệu € thay vì €
    features = ['Age', 'Overall', 'Potential', 'Wage_Numeric']
    
    print("=" * 60)
    print("DEMO CÂY HỒI QUY (REGRESSION TREE)")
    print("Dự đoán giá trị cầu thủ (đơn vị: triệu €)")
    print("=" * 60)
    print(f"\nSố lượng mẫu: {len(df_binned)}")
    print(f"Features: {features}")
    print(f"Target: {target_column}")
    print(f"\nThống kê giá trị (triệu €):")
    print(f"  Min: {df_binned[target_column].min():.2f}M€")
    print(f"  Max: {df_binned[target_column].max():.2f}M€")
    print(f"  Mean: {df_binned[target_column].mean():.2f}M€")
    print("\nDữ liệu mẫu:")
    print(df_binned[features + [target_column]].head(10).to_string())
    
    # Tạo dataframe cho training
    train_df = df_binned[features + [target_column]].copy()
    
    # Test với MSE
    print("\n" + "=" * 60)
    print("1. CÂY HỒI QUY VỚI MSE")
    print("=" * 60)
    tree_mse = RegressionTree(criterion='mse', max_depth=3)
    tree_mse.fit(train_df, target_column)
    print("\nCấu trúc cây:")
    tree_mse.print_tree()
    
    r2 = tree_mse.score(train_df, target_column)
    mse = tree_mse.mse_score(train_df, target_column)
    rmse = np.sqrt(mse)
    mae = tree_mse.mae_score(train_df, target_column)
    
    # Tính MAPE (Mean Absolute Percentage Error)
    predictions = tree_mse.predict(train_df)
    actual = train_df[target_column].values
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    print(f"\n📊 Kết quả Regression với MSE:")
    print(f"  R² Score: {r2:.4f} (giải thích {r2*100:.1f}% variance)")
    print(f"  RMSE: {rmse:.2f} triệu € (sai số trung bình)")
    print(f"  MAE: {mae:.2f} triệu €") 
    print(f"  MAPE: {mape:.1f}% (sai số phần trăm)")
    
    # Test với MAE
    print("\n" + "=" * 60)
    print("2. CÂY HỒI QUY VỚI MAE")
    print("=" * 60)
    tree_mae = RegressionTree(criterion='mae', max_depth=3)
    tree_mae.fit(train_df, target_column)
    print("\nCấu trúc cây:")
    tree_mae.print_tree()
    
    r2_mae = tree_mae.score(train_df, target_column)
    mse_mae = tree_mae.mse_score(train_df, target_column)
    rmse_mae = np.sqrt(mse_mae)
    mae_val = tree_mae.mae_score(train_df, target_column)
    
    predictions_mae_tree = tree_mae.predict(train_df)
    mape_mae = np.mean(np.abs((actual - predictions_mae_tree) / actual)) * 100
    
    print(f"\n📊 Kết quả Regression với MAE:")
    print(f"  R² Score: {r2_mae:.4f} (giải thích {r2_mae*100:.1f}% variance)")
    print(f"  RMSE: {rmse_mae:.2f} triệu €")
    print(f"  MAE: {mae_val:.2f} triệu €")
    print(f"  MAPE: {mape_mae:.1f}%")
    
    # Test dự đoán
    print("\n" + "=" * 60)
    print("3. DỰ ĐOÁN MẪU MỚI")
    print("=" * 60)
    test_data = pd.DataFrame({
        'Age': ['Young', 'Prime', 'Experienced', 'Veteran'],
        'Overall': ['Elite', 'High', 'Medium', 'Low'],
        'Potential': ['Elite', 'High', 'Medium', 'Low'],
        'Wage_Numeric': ['Elite', 'High', 'Medium', 'Low']
    })
    print("\nDữ liệu test (4 cầu thủ mẫu):")
    print(test_data)
    
    predictions_mse = tree_mse.predict(test_data)
    predictions_mae = tree_mae.predict(test_data)
    
    print("\nKết quả dự đoán giá trị cầu thủ:")
    print(f"MSE criterion: {[f'{p:.1f}M€' for p in predictions_mse]}")
    print(f"MAE criterion: {[f'{p:.1f}M€' for p in predictions_mae]}")
    print("=" * 60)

