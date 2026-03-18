"""
Module 1: Data Cleaning
- Xử lý duplicates
- Xóa cột không cần thiết
- Xóa cột có variance thấp
- Phát hiện multicollinearity
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Class để làm sạch dữ liệu"""
    
    def __init__(self, 
                 missing_threshold: float = 0.5,
                 low_variance_threshold: float = 0.01,
                 high_corr_threshold: float = 0.95):
        """
        Parameters:
        -----------
        missing_threshold : float
            Ngưỡng để xóa cột (cột có >threshold missing sẽ bị xóa)
        low_variance_threshold : float
            Ngưỡng variance thấp
        high_corr_threshold : float
            Ngưỡng correlation cao
        """
        self.missing_threshold = missing_threshold
        self.low_variance_threshold = low_variance_threshold
        self.high_corr_threshold = high_corr_threshold
        self.dropped_columns = []
        self.report = {}
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xóa dữ liệu trùng lặp"""
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        
        print(f"Số hàng ban đầu: {initial_rows}")
        print(f"Số hàng trùng lặp: {duplicates_removed}")
        print(f"Số hàng còn lại: {len(df_clean)}")
        
        self.report['duplicates_removed'] = duplicates_removed
        
        return df_clean
    
    def remove_unnecessary_columns(self, 
                                   df: pd.DataFrame, 
                                   columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
        """Xóa các cột không cần thiết (URL, ID, ...)"""
        print("\n" + "="*70)
        print("2. XÓA CỘT KHÔNG CẦN THIẾT")
        print("="*70)
        
        if columns_to_drop is None:
            columns_to_drop = []
        
        # Tự động phát hiện cột URL, ID
        auto_drop = []
        identifier_keywords = ['url', 'id', '_id']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in identifier_keywords):
                auto_drop.append(col)
        
        # Kết hợp với danh sách do user cung cấp
        all_drop = list(set(columns_to_drop + auto_drop))
        
        # Chỉ xóa các cột tồn tại
        existing_drop = [col for col in all_drop if col in df.columns]
        
        if existing_drop:
            print(f"Xóa {len(existing_drop)} cột:")
            for col in existing_drop:
                print(f"  - {col}")
            
            df_clean = df.drop(columns=existing_drop)
            self.dropped_columns.extend(existing_drop)
        else:
            print("Không có cột nào cần xóa")
            df_clean = df.copy()
        
        self.report['unnecessary_columns_removed'] = len(existing_drop)
        
        return df_clean
    
    def remove_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xóa cột có quá nhiều giá trị thiếu"""
        print("\n" + "="*70)
        print("3. XÓA CỘT CÓ QUÁ NHIỀU MISSING VALUES")
        print("="*70)
        
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        
        if cols_to_drop:
            print(f"Xóa {len(cols_to_drop)} cột có >{self.missing_threshold*100}% missing:")
            for col in cols_to_drop:
                print(f"  - {col}: {missing_ratios[col]*100:.2f}%")
            
            df_clean = df.drop(columns=cols_to_drop)
            self.dropped_columns.extend(cols_to_drop)
        else:
            print(f"Không có cột nào có >{self.missing_threshold*100}% missing")
            df_clean = df.copy()
        
        self.report['high_missing_columns_removed'] = len(cols_to_drop)
        
        return df_clean
    
    def remove_low_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xóa cột có variance thấp"""
        print("\n" + "="*70)
        print("4. XÓA CỘT CÓ VARIANCE THẤP")
        print("="*70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_var_cols = []
        
        for col in numeric_cols:
            var = df[col].var()
            if var < self.low_variance_threshold or df[col].nunique() == 1:
                low_var_cols.append(col)
                print(f"  - {col}: var={var:.6f}, unique={df[col].nunique()}")
        
        if low_var_cols:
            print(f"\nXóa {len(low_var_cols)} cột có variance thấp")
            df_clean = df.drop(columns=low_var_cols)
            self.dropped_columns.extend(low_var_cols)
        else:
            print("Không có cột nào có variance quá thấp")
            df_clean = df.copy()
        
        self.report['low_variance_columns_removed'] = len(low_var_cols)
        
        return df_clean
    
    def detect_multicollinearity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple]]:
        """Phát hiện các cặp cột có correlation cao"""
        print("\n" + "="*70)
        print("5. PHÁT HIỆN MULTICOLLINEARITY")
        print("="*70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Không đủ cột số để tính correlation")
            return df, []
        
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Tìm các cặp có correlation cao
        high_corr_pairs = []
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        for col in upper_triangle.columns:
            high_corr = upper_triangle[col][upper_triangle[col] > self.high_corr_threshold]
            for idx in high_corr.index:
                high_corr_pairs.append((col, idx, high_corr[idx]))
        
        if high_corr_pairs:
            print(f"Tìm thấy {len(high_corr_pairs)} cặp cột có correlation >{self.high_corr_threshold}:")
            for col1, col2, corr_val in high_corr_pairs[:10]:
                print(f"  - {col1} <-> {col2}: {corr_val:.3f}")
            
            if len(high_corr_pairs) > 10:
                print(f"  ... và {len(high_corr_pairs)-10} cặp khác")
        else:
            print(f"Không có cặp nào có correlation >{self.high_corr_threshold}")
        
        self.report['high_correlation_pairs'] = len(high_corr_pairs)
        
        return df, high_corr_pairs
    
    def remove_high_corr_columns(self, 
                                df: pd.DataFrame, 
                                high_corr_pairs: List[Tuple]) -> pd.DataFrame:
        """Xóa một trong hai cột có correlation cao (giữ cột đầu tiên)"""
        if not high_corr_pairs:
            return df
        
        cols_to_drop = set()
        for col1, col2, _ in high_corr_pairs:
            # Giữ col1, xóa col2
            cols_to_drop.add(col2)
        
        cols_to_drop = list(cols_to_drop)
        
        if cols_to_drop:
            print(f"\nXóa {len(cols_to_drop)} cột có correlation cao:")
            for col in cols_to_drop[:10]:
                print(f"  - {col}")
            if len(cols_to_drop) > 10:
                print(f"  ... và {len(cols_to_drop)-10} cột khác")
            
            df_clean = df.drop(columns=cols_to_drop)
            self.dropped_columns.extend(cols_to_drop)
            self.report['high_corr_columns_removed'] = len(cols_to_drop)
        else:
            df_clean = df.copy()
            self.report['high_corr_columns_removed'] = 0
        
        return df_clean
    
    def clean(self, 
             df: pd.DataFrame, 
             columns_to_drop: Optional[List[str]] = None,
             remove_high_corr: bool = False) -> pd.DataFrame:
        """
        Thực hiện toàn bộ quá trình cleaning
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần clean
        columns_to_drop : List[str], optional
            Danh sách cột cần xóa thủ công
        remove_high_corr : bool
            Có xóa cột có correlation cao không
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã được clean
        """
        print("\n" + "="*70)
        print("BẮT ĐẦU DATA CLEANING")
        print("="*70)
        print(f"Kích thước ban đầu: {df.shape}")
        
        # 1. Xóa duplicates
        df_clean = self.remove_duplicates(df)
        
        # 2. Xóa cột không cần thiết
        df_clean = self.remove_unnecessary_columns(df_clean, columns_to_drop)
        
        # 3. Xóa cột có nhiều missing
        df_clean = self.remove_high_missing_columns(df_clean)
        
        # 4. Xóa cột có variance thấp
        df_clean = self.remove_low_variance_columns(df_clean)
        
        # 5. Phát hiện multicollinearity
        df_clean, high_corr_pairs = self.detect_multicollinearity(df_clean)
        
        # 6. Xóa cột có correlation cao (optional)
        if remove_high_corr:
            df_clean = self.remove_high_corr_columns(df_clean, high_corr_pairs)
        
        print("\n" + "="*70)
        print("KẾT QUẢ DATA CLEANING")
        print("="*70)
        print(f"Kích thước ban đầu: {df.shape}")
        print(f"Kích thước sau cleaning: {df_clean.shape}")
        print(f"Đã xóa {len(self.dropped_columns)} cột")
        print(f"Đã xóa {self.report.get('duplicates_removed', 0)} hàng trùng lặp")
        
        return df_clean
    
    def get_report(self) -> dict:
        """Lấy báo cáo chi tiết về quá trình cleaning"""
        return self.report


if __name__ == "__main__":
    # Test
    df = pd.read_csv('../sofifa_players.csv')
    
    cleaner = DataCleaner(
        missing_threshold=0.5,
        low_variance_threshold=0.01,
        high_corr_threshold=0.95
    )
    
    df_clean = cleaner.clean(
        df, 
        columns_to_drop=['Name', 'Team', 'Nationality'],
        remove_high_corr=False
    )
    
    print("\n" + cleaner.get_report().__str__())
    
    # Save
    df_clean.to_csv('../data/cleaned_data.csv', index=False)
    print("\n✓ Saved to data/cleaned_data.csv")
