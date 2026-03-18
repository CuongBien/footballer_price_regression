"""
Multi-Label Encoding
Xử lý các cột có nhiều giá trị categorical trong 1 cell
Ví dụ: Positions = "CM, CDM, CAM" → Positions_CM=1, Positions_CDM=1, Positions_CAM=1, Positions_ST=0, ...
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class MultiLabelEncoder:
    """Encode multi-label categorical columns thành binary columns"""
    
    def __init__(self, delimiter: str = ', '):
        """
        Parameters:
        -----------
        delimiter : str
            Ký tự ngăn cách giữa các labels (mặc định: ', ')
        """
        self.delimiter = delimiter
        self.label_encoders = {}  # {column_name: [label1, label2, ...]}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, columns: List[str]):
        """
        Học tất cả unique labels từ data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data để fit
        columns : List[str]
            Danh sách các cột cần encode
        """
        print("\nFitting MultiLabelEncoder...")
        
        for col in columns:
            if col not in df.columns:
                print(f"  ⚠️  Column '{col}' không tồn tại, bỏ qua")
                continue
            
            # Lấy tất cả unique labels
            all_labels = set()
            
            for value in df[col].dropna():
                if isinstance(value, str):
                    # Split và strip whitespace
                    labels = [label.strip() for label in value.split(self.delimiter)]
                    all_labels.update(labels)
            
            # Loại bỏ empty strings và 'SUB'
            all_labels = {label for label in all_labels if label and label != 'SUB'}
            
            self.label_encoders[col] = sorted(all_labels)
            
            print(f"  ✓ Column '{col}': {len(all_labels)} unique labels")
            if len(all_labels) <= 20:
                print(f"    Labels: {', '.join(sorted(all_labels))}")
            else:
                print(f"    Labels: {', '.join(sorted(all_labels)[:20])} ... ({len(all_labels)-20} more)")
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform multi-label columns thành binary columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data cần transform
            
        Returns:
        --------
        pd.DataFrame
            Data với binary columns thay cho multi-label columns
        """
        if not self.fitted:
            raise ValueError("MultiLabelEncoder chưa được fit! Gọi fit() trước.")
        
        df_transformed = df.copy()
        
        for col, labels in self.label_encoders.items():
            if col not in df_transformed.columns:
                print(f"  ⚠️  Column '{col}' không có trong data, bỏ qua")
                continue
            
            # Tạo binary columns cho mỗi label
            for label in labels:
                new_col_name = f"{col}_{label}"
                
                # Check nếu cell chứa label này
                df_transformed[new_col_name] = df_transformed[col].apply(
                    lambda x: 1 if isinstance(x, str) and label in [l.strip() for l in x.split(self.delimiter)] else 0
                )
            
            # Drop original column
            df_transformed = df_transformed.drop(columns=[col])
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit và transform trong 1 bước"""
        self.fit(df, columns)
        return self.transform(df)
    
    def get_feature_names(self, column: str) -> List[str]:
        """Lấy tên các binary columns được tạo ra từ 1 column"""
        if column not in self.label_encoders:
            return []
        
        return [f"{column}_{label}" for label in self.label_encoders[column]]
    
    def get_all_feature_names(self) -> List[str]:
        """Lấy tên tất cả binary columns"""
        all_features = []
        for col in self.label_encoders:
            all_features.extend(self.get_feature_names(col))
        return all_features


def detect_multi_label_columns(df: pd.DataFrame, 
                               delimiter: str = ',',
                               min_samples: int = 5) -> List[str]:
    """
    Tự động phát hiện các cột có multi-label
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data cần kiểm tra
    delimiter : str
        Delimiter để split
    min_samples : int
        Số sample tối thiểu có delimiter để coi là multi-label
        
    Returns:
    --------
    List[str]
        Danh sách các cột có multi-label
    """
    multi_label_cols = []
    
    for col in df.columns:
        # Chỉ check string columns
        if df[col].dtype == 'object':
            # Đếm số samples có delimiter
            count_with_delimiter = df[col].astype(str).str.contains(delimiter, na=False).sum()
            
            if count_with_delimiter >= min_samples:
                multi_label_cols.append(col)
    
    return multi_label_cols


# Example usage
if __name__ == "__main__":
    # Test data
    data = {
        'Name': ['Player1', 'Player2', 'Player3'],
        'Positions': ['ST, CF', 'CM, CDM, CAM', 'RW, LW'],
    }
    
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    # Encode
    encoder = MultiLabelEncoder(delimiter=', ')
    df_encoded = encoder.fit_transform(df, ['Positions'])
    
    print("\nEncoded data:")
    print(df_encoded)
    
    print("\nNew feature names:")
    print(encoder.get_all_feature_names())
