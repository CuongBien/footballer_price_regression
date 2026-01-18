"""
Module 4: Encoding Categorical Variables
- Label Encoding
- One-Hot Encoding
- Target Encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class Encoder:
    """Class để encode các biến phân loại"""
    
    def __init__(self,
                 method: str = 'onehot',
                 max_categories: int = 10,
                 handle_unknown: str = 'ignore'):
        """
        Parameters:
        -----------
        method : str
            Phương pháp encoding: 'onehot', 'label', 'target'
        max_categories : int
            Số categories tối đa cho OneHot (nếu >max_categories sẽ dùng Label)
        handle_unknown : str
            Cách xử lý categories mới: 'ignore', 'error'
        """
        self.method = method
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.encoders = {}
        self.encoded_columns = []
        self.onehot_columns = []
        
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'Encoder':
        """
        Fit encoder
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame để fit
        target : pd.Series, optional
            Target variable (cần cho target encoding)
            
        Returns:
        --------
        self
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = ('label', encoder)
                
            elif self.method == 'onehot':
                if n_unique <= self.max_categories:
                    # Dùng OneHot
                    encoder = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown=self.handle_unknown
                    )
                    encoder.fit(df[[col]])
                    self.encoders[col] = ('onehot', encoder)
                    self.onehot_columns.append(col)
                else:
                    # Quá nhiều categories, dùng Label
                    print(f"  ! {col} có {n_unique} categories (>{self.max_categories}), dùng Label Encoding")
                    encoder = LabelEncoder()
                    encoder.fit(df[col].astype(str))
                    self.encoders[col] = ('label', encoder)
                    
            elif self.method == 'target':
                if target is not None:
                    # Target encoding: mean của target cho mỗi category
                    target_means = df.groupby(col)[target.name].mean().to_dict()
                    self.encoders[col] = ('target', target_means)
                else:
                    # Fallback to label encoding
                    print(f"  ! Target not provided, using Label Encoding for {col}")
                    encoder = LabelEncoder()
                    encoder.fit(df[col].astype(str))
                    self.encoders[col] = ('label', encoder)
        
        self.encoded_columns = list(categorical_cols)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần transform
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã được encode
        """
        df_encoded = df.copy()
        
        for col, (method, encoder) in self.encoders.items():
            if col not in df_encoded.columns:
                continue
                
            if method == 'label':
                # Label encoding
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                except ValueError:
                    # Handle unknown categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0] 
                        if str(x) in encoder.classes_ 
                        else -1
                    )
                    
            elif method == 'onehot':
                # One-hot encoding
                encoded = encoder.transform(df_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=feature_names,
                    index=df_encoded.index
                )
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                
            elif method == 'target':
                # Target encoding
                target_means = encoder
                df_encoded[col] = df_encoded[col].map(target_means)
                # Fill unknown categories với global mean
                global_mean = np.mean(list(target_means.values()))
                df_encoded[col] = df_encoded[col].fillna(global_mean)
        
        return df_encoded
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit và transform trong một bước
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        target : pd.Series, optional
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã được encode
        """
        print("\n" + "="*70)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*70)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            print("\nKhông có cột phân loại nào! Bỏ qua bước encoding.")
            return df
        
        print(f"\nSố cột phân loại: {len(categorical_cols)}")
        print(f"Phương pháp: {self.method}")
        
        print(f"\nCác cột phân loại:")
        for col in categorical_cols:
            n_unique = df[col].nunique()
            print(f"  - {col}: {n_unique} categories")
        
        # Fit và transform
        self.fit(df, target)
        df_encoded = self.transform(df)
        
        # Báo cáo
        print(f"\n✓ Đã encode {len(self.encoded_columns)} cột")
        print(f"✓ Kích thước trước: {df.shape}")
        print(f"✓ Kích thước sau: {df_encoded.shape}")
        
        if self.onehot_columns:
            print(f"\n✓ OneHot Encoding cho: {self.onehot_columns}")
        
        return df_encoded
    
    def get_encoding_info(self) -> Dict:
        """Lấy thông tin về encoding"""
        info = {
            'method': self.method,
            'encoded_columns': self.encoded_columns,
            'onehot_columns': self.onehot_columns,
            'n_encoded': len(self.encoded_columns)
        }
        return info


if __name__ == "__main__":
    # Test
    df = pd.read_csv('../data/imputed_data.csv')
    
    encoder = Encoder(
        method='onehot',
        max_categories=10
    )
    
    df_encoded = encoder.fit_transform(df)
    
    print("\n" + str(encoder.get_encoding_info()))
    
    # Save
    df_encoded.to_csv('../data/encoded_data.csv', index=False)
    print("\n✓ Saved to data/encoded_data.csv")
