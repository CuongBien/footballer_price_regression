"""
Module 5: Scaling/Normalization
- Standard Scaling (Z-score)
- Min-Max Scaling
- Robust Scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class Scaler:
    """Class để scale dữ liệu số"""
    
    def __init__(self, method: str = 'standard'):
        """
        Parameters:
        -----------
        method : str
            Phương pháp scaling: 'standard', 'minmax', 'robust'
        """
        self.method = method
        self.scaler = None
        self.scaled_columns = []
        
        # Khởi tạo scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> 'Scaler':
        """
        Fit scaler
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame để fit
        exclude_cols : List[str], optional
            Các cột không cần scale (vd: target, binary columns)
            
        Returns:
        --------
        self
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        self.scaled_columns = numeric_cols
        
        if len(numeric_cols) > 0:
            self.scaler.fit(df[numeric_cols])
        
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
            DataFrame đã được scale
        """
        df_scaled = df.copy()
        
        if len(self.scaled_columns) > 0:
            scaled_data = self.scaler.transform(df_scaled[self.scaled_columns])
            df_scaled[self.scaled_columns] = scaled_data
        
        return df_scaled
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit và transform trong một bước
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        exclude_cols : List[str], optional
            Các cột không cần scale
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã được scale
        """
        print("\n" + "="*70)
        print("SCALING/NORMALIZATION")
        print("="*70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("\nKhông có cột số nào! Bỏ qua bước scaling.")
            return df
        
        print(f"\nPhương pháp: {self.method.upper()}")
        print(f"Số cột số: {len(numeric_cols)}")
        
        if exclude_cols:
            print(f"\nCác cột bỏ qua scaling: {exclude_cols}")
        
        # Fit và transform
        self.fit(df, exclude_cols)
        df_scaled = self.transform(df)
        
        # Báo cáo thống kê trước và sau
        print(f"\n✓ Đã scale {len(self.scaled_columns)} cột")
        
        print("\nThống kê trước scaling (5 cột đầu):")
        for col in self.scaled_columns[:5]:
            print(f"  - {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
        
        print("\nThống kê sau scaling (5 cột đầu):")
        for col in self.scaled_columns[:5]:
            print(f"  - {col}: mean={df_scaled[col].mean():.2f}, std={df_scaled[col].std():.2f}")
        
        return df_scaled
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển dữ liệu đã scale về dạng gốc
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame đã được scale
            
        Returns:
        --------
        pd.DataFrame
            DataFrame ở dạng gốc
        """
        df_original = df.copy()
        
        if len(self.scaled_columns) > 0:
            original_data = self.scaler.inverse_transform(df_original[self.scaled_columns])
            df_original[self.scaled_columns] = original_data
        
        return df_original
    
    def get_scaling_params(self) -> dict:
        """Lấy các tham số scaling"""
        params = {
            'method': self.method,
            'scaled_columns': self.scaled_columns,
            'n_scaled': len(self.scaled_columns)
        }
        
        if self.method == 'standard':
            params['means'] = dict(zip(self.scaled_columns, self.scaler.mean_))
            params['stds'] = dict(zip(self.scaled_columns, self.scaler.scale_))
        elif self.method == 'minmax':
            params['mins'] = dict(zip(self.scaled_columns, self.scaler.data_min_))
            params['maxs'] = dict(zip(self.scaled_columns, self.scaler.data_max_))
        
        return params


if __name__ == "__main__":
    # Test
    df = pd.read_csv('../data/encoded_data.csv')
    
    scaler = Scaler(method='standard')
    
    # Giả sử 'Overall' là target, không scale
    df_scaled = scaler.fit_transform(df, exclude_cols=['Overall'])
    
    print("\n" + str(scaler.get_scaling_params()))
    
    # Save
    df_scaled.to_csv('../data/scaled_data.csv', index=False)
    print("\n✓ Saved to data/scaled_data.csv")
