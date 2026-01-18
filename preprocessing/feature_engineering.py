"""
Module 6: Feature Engineering & Selection
- Tạo features mới
- Feature Selection
- Polynomial Features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    f_regression, 
    f_classif,
    RFE
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Class để tạo và chọn features"""
    
    def __init__(self,
                 create_polynomial: bool = False,
                 polynomial_degree: int = 2,
                 selection_method: Optional[str] = None,
                 n_features: Optional[int] = None):
        """
        Parameters:
        -----------
        create_polynomial : bool
            Có tạo polynomial features không
        polynomial_degree : int
            Bậc của polynomial
        selection_method : str, optional
            Phương pháp chọn features: 'variance', 'correlation', 'model_based', None
        n_features : int, optional
            Số features muốn giữ lại (None = giữ tất cả)
        """
        self.create_polynomial = create_polynomial
        self.polynomial_degree = polynomial_degree
        self.selection_method = selection_method
        self.n_features = n_features
        self.poly_transformer = None
        self.selector = None
        self.selected_features = []
        self.feature_importances = {}
        
    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo polynomial features"""
        print("\nTạo Polynomial Features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("Không có cột số để tạo polynomial features")
            return df
        
        self.poly_transformer = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=False
        )
        
        poly_features = self.poly_transformer.fit_transform(df[numeric_cols])
        poly_feature_names = self.poly_transformer.get_feature_names_out(numeric_cols)
        
        df_poly = pd.DataFrame(
            poly_features,
            columns=poly_feature_names,
            index=df.index
        )
        
        # Xóa các cột gốc và thêm polynomial features
        df_new = df.drop(columns=numeric_cols)
        df_new = pd.concat([df_new, df_poly], axis=1)
        
        print(f"✓ Tạo {len(poly_feature_names)} polynomial features")
        print(f"✓ Kích thước mới: {df_new.shape}")
        
        return df_new
    
    def select_features_by_variance(self, 
                                   df: pd.DataFrame, 
                                   threshold: float = 0.01) -> pd.DataFrame:
        """Chọn features dựa trên variance"""
        print("\nChọn features theo Variance...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[numeric_cols])
        
        selected_cols = [col for col, selected in zip(numeric_cols, selector.get_support()) 
                        if selected]
        
        print(f"✓ Giữ {len(selected_cols)}/{len(numeric_cols)} features")
        
        self.selected_features = selected_cols
        
        # Giữ các cột được chọn + các cột không phải số
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return df[selected_cols + non_numeric_cols]
    
    def select_features_by_correlation(self, 
                                      df: pd.DataFrame, 
                                      target_col: str,
                                      n_features: Optional[int] = None) -> pd.DataFrame:
        """Chọn features dựa trên correlation với target"""
        print("\nChọn features theo Correlation với target...")
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' không tồn tại!")
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        # Tính correlation với target
        correlations = df[numeric_cols + [target_col]].corr()[target_col].abs()
        correlations = correlations.drop(target_col).sort_values(ascending=False)
        
        # Chọn top N features
        if n_features:
            selected_cols = correlations.head(n_features).index.tolist()
        else:
            # Chọn features có correlation > 0.1
            selected_cols = correlations[correlations > 0.1].index.tolist()
        
        print(f"✓ Giữ {len(selected_cols)}/{len(numeric_cols)} features")
        print("\nTop 10 features có correlation cao nhất:")
        for col, corr in correlations.head(10).items():
            print(f"  - {col}: {corr:.3f}")
        
        self.selected_features = selected_cols
        self.feature_importances = correlations.to_dict()
        
        # Giữ target và các features được chọn
        return df[selected_cols + [target_col]]
    
    def select_features_by_model(self, 
                                df: pd.DataFrame, 
                                target_col: str,
                                task_type: str = 'regression',
                                n_features: Optional[int] = None) -> pd.DataFrame:
        """Chọn features dựa trên feature importance từ model"""
        print("\nChọn features bằng Model-Based Selection...")
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' không tồn tại!")
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Chọn model phù hợp
        if task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Train model
        model.fit(X, y)
        
        # Lấy feature importances
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        
        # Chọn top N features
        if n_features:
            selected_cols = importances.head(n_features).index.tolist()
        else:
            # Chọn features có importance > mean
            threshold = importances.mean()
            selected_cols = importances[importances > threshold].index.tolist()
        
        print(f"✓ Giữ {len(selected_cols)}/{len(feature_cols)} features")
        print("\nTop 10 features quan trọng nhất:")
        for col, imp in importances.head(10).items():
            print(f"  - {col}: {imp:.4f}")
        
        self.selected_features = selected_cols
        self.feature_importances = importances.to_dict()
        
        # Giữ target và các features được chọn
        return df[selected_cols + [target_col]]
    
    def engineer(self, 
                df: pd.DataFrame, 
                target_col: Optional[str] = None,
                task_type: str = 'regression') -> pd.DataFrame:
        """
        Thực hiện toàn bộ feature engineering
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần xử lý
        target_col : str, optional
            Tên cột target (cần cho feature selection)
        task_type : str
            Loại task: 'regression' hoặc 'classification'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã được engineer
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING & SELECTION")
        print("="*70)
        print(f"Kích thước ban đầu: {df.shape}")
        
        df_engineered = df.copy()
        
        # 1. Tạo polynomial features (nếu được yêu cầu)
        if self.create_polynomial:
            df_engineered = self.create_polynomial_features(df_engineered)
        
        # 2. Feature selection (nếu được yêu cầu)
        if self.selection_method:
            if self.selection_method == 'variance':
                df_engineered = self.select_features_by_variance(df_engineered)
            
            elif self.selection_method == 'correlation':
                if target_col:
                    df_engineered = self.select_features_by_correlation(
                        df_engineered, 
                        target_col, 
                        self.n_features
                    )
                else:
                    print("⚠️  Cần target_col cho correlation-based selection")
            
            elif self.selection_method == 'model_based':
                if target_col:
                    df_engineered = self.select_features_by_model(
                        df_engineered, 
                        target_col, 
                        task_type,
                        self.n_features
                    )
                else:
                    print("⚠️  Cần target_col cho model-based selection")
        
        print(f"\n✓ Kích thước cuối cùng: {df_engineered.shape}")
        
        return df_engineered
    
    def get_feature_report(self) -> dict:
        """Lấy báo cáo về features"""
        return {
            'selected_features': self.selected_features,
            'n_selected': len(self.selected_features),
            'feature_importances': self.feature_importances
        }


if __name__ == "__main__":
    # Test
    df = pd.read_csv('../data/scaled_data.csv')
    
    engineer = FeatureEngineer(
        create_polynomial=False,
        selection_method='correlation',
        n_features=20
    )
    
    df_engineered = engineer.engineer(
        df, 
        target_col='Overall',
        task_type='regression'
    )
    
    print("\n" + str(engineer.get_feature_report()))
    
    # Save
    df_engineered.to_csv('../data/engineered_data.csv', index=False)
    print("\n✓ Saved to data/engineered_data.csv")
