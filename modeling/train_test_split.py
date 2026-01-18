"""
Module 2: Train/Test Split
- Chia dữ liệu thành train/validation/test
- Hỗ trợ stratify cho classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataSplitter:
    """Class để chia dữ liệu train/validation/test"""
    
    def __init__(self,
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 random_state: int = 42,
                 stratify_column: Optional[str] = None):
        """
        Parameters:
        -----------
        test_size : float
            Tỷ lệ test set (0-1)
        validation_size : float
            Tỷ lệ validation set từ train set (0-1)
        random_state : int
            Random seed
        stratify_column : str, optional
            Cột để stratify (cho classification)
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify_column = stratify_column
        self.split_info = {}
        
    def split(self, 
             df: pd.DataFrame, 
             target_col: str,
             include_validation: bool = True) -> Tuple:
        """
        Chia dữ liệu thành train/validation/test
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần chia
        target_col : str
            Tên cột target
        include_validation : bool
            Có chia validation set không
            
        Returns:
        --------
        Tuple : (X_train, X_val, X_test, y_train, y_val, y_test) nếu include_validation=True
                (X_train, X_test, y_train, y_test) nếu include_validation=False
        """
        print("\n" + "="*70)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("="*70)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' không tồn tại trong DataFrame")
        
        # Tách X và y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nKích thước dữ liệu: {df.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Target: {target_col}")
        
        # Xử lý stratify
        stratify_train_test = None
        if self.stratify_column:
            if self.stratify_column == target_col:
                stratify_train_test = y
                print(f"\n✓ Stratify theo target: {target_col}")
            elif self.stratify_column in df.columns:
                stratify_train_test = df[self.stratify_column]
                print(f"\n✓ Stratify theo: {self.stratify_column}")
        
        # Chia train + validation và test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_train_test
        )
        
        print(f"\nTest size: {self.test_size*100:.1f}%")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        
        self.split_info['test_size'] = len(X_test)
        self.split_info['test_ratio'] = self.test_size
        
        if include_validation and self.validation_size > 0:
            # Chia train và validation
            stratify_train_val = None
            if self.stratify_column and self.stratify_column == target_col:
                stratify_train_val = y_temp
            
            # Tính validation size từ temp set
            val_size_adjusted = self.validation_size / (1 - self.test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=stratify_train_val
            )
            
            print(f"\nValidation size: {self.validation_size*100:.1f}%")
            print(f"  - X_val: {X_val.shape}")
            print(f"  - y_val: {y_val.shape}")
            
            print(f"\nTrain size: {(1-self.test_size-self.validation_size)*100:.1f}%")
            print(f"  - X_train: {X_train.shape}")
            print(f"  - y_train: {y_train.shape}")
            
            self.split_info['train_size'] = len(X_train)
            self.split_info['val_size'] = len(X_val)
            self.split_info['train_ratio'] = 1 - self.test_size - self.validation_size
            self.split_info['val_ratio'] = self.validation_size
            
            # Thống kê target distribution
            print("\nPhân phối target:")
            print(f"  - Train: {y_train.describe()}")
            print(f"  - Validation: {y_val.describe()}")
            print(f"  - Test: {y_test.describe()}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Không có validation set
            X_train = X_temp
            y_train = y_temp
            
            print(f"\nTrain size: {(1-self.test_size)*100:.1f}%")
            print(f"  - X_train: {X_train.shape}")
            print(f"  - y_train: {y_train.shape}")
            
            self.split_info['train_size'] = len(X_train)
            self.split_info['train_ratio'] = 1 - self.test_size
            
            # Thống kê target distribution
            print("\nPhân phối target:")
            print(f"  - Train: {y_train.describe()}")
            print(f"  - Test: {y_test.describe()}")
            
            return X_train, X_test, y_train, y_test
    
    def get_split_info(self) -> dict:
        """Lấy thông tin về split"""
        return self.split_info


if __name__ == "__main__":
    # Test
    df = pd.read_csv('../data/engineered_data.csv')
    
    splitter = DataSplitter(
        test_size=0.2,
        validation_size=0.1,
        random_state=42,
        stratify_column=None  # Hoặc 'Overall' cho classification
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        df, 
        target_col='Overall',
        include_validation=True
    )
    
    print("\n" + str(splitter.get_split_info()))
    
    # Save splits
    pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
    pd.DataFrame(X_val).to_csv('../data/X_val.csv', index=False)
    pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
    pd.DataFrame(y_val).to_csv('../data/y_val.csv', index=False)
    pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
    
    print("\n✓ Saved train/val/test splits to data/")
