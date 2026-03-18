"""
Module 7: Handle Imbalanced Data
- SMOTE
- ADASYN
- Random Over/Under Sampling
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. Install: pip install imbalanced-learn")


class ImbalanceHandler:
    """Class để xử lý dữ liệu mất cân bằng (cho classification)"""
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Parameters:
        -----------
        method : str
            Phương pháp: 'smote', 'adasyn', 'random_oversample', 'random_undersample'
        random_state : int
            Random seed
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError("Cần cài đặt imbalanced-learn: pip install imbalanced-learn")
        
        self.method = method
        self.random_state = random_state
        self.sampler = None
        self.original_distribution = {}
        self.resampled_distribution = {}
        
        # Khởi tạo sampler
        if method == 'smote':
            self.sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            self.sampler = ADASYN(random_state=random_state)
        elif method == 'random_oversample':
            self.sampler = RandomOverSampler(random_state=random_state)
        elif method == 'random_undersample':
            self.sampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resample dữ liệu để cân bằng
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            X_resampled, y_resampled
        """
        print("\n" + "="*70)
        print("HANDLE IMBALANCED DATA")
        print("="*70)
        
        print(f"\nPhương pháp: {self.method.upper()}")
        print(f"Kích thước ban đầu: {X.shape}")
        
        # Lưu phân phối ban đầu
        self.original_distribution = y.value_counts().to_dict()
        print("\nPhân phối ban đầu:")
        for label, count in sorted(self.original_distribution.items()):
            print(f"  - Class {label}: {count} ({count/len(y)*100:.2f}%)")
        
        # Resample
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        # Convert về DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Lưu phân phối sau resample
        self.resampled_distribution = y_resampled.value_counts().to_dict()
        
        print(f"\nKích thước sau resample: {X_resampled.shape}")
        print("\nPhân phối sau resample:")
        for label, count in sorted(self.resampled_distribution.items()):
            print(f"  - Class {label}: {count} ({count/len(y_resampled)*100:.2f}%)")
        
        return X_resampled, y_resampled
    
    def get_resample_info(self) -> dict:
        """Lấy thông tin về resampling"""
        return {
            'method': self.method,
            'original_distribution': self.original_distribution,
            'resampled_distribution': self.resampled_distribution
        }


if __name__ == "__main__":
    # Test (chỉ dùng cho classification với imbalanced data)
    if IMBLEARN_AVAILABLE:
        X_train = pd.read_csv('../data/X_train.csv')
        y_train = pd.read_csv('../data/y_train.csv').squeeze()
        
        # Kiểm tra xem có imbalanced không
        distribution = y_train.value_counts()
        print(f"Class distribution:\n{distribution}")
        
        if len(distribution) > 1:
            handler = ImbalanceHandler(method='smote', random_state=42)
            X_resampled, y_resampled = handler.resample(X_train, y_train)
            
            print("\n" + str(handler.get_resample_info()))
            
            # Save
            X_resampled.to_csv('../data/X_train_resampled.csv', index=False)
            y_resampled.to_csv('../data/y_train_resampled.csv', index=False)
            print("\n✓ Saved resampled data")
        else:
            print("\n⚠️  Regression task - không cần resample")
