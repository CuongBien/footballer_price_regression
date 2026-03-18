"""
DATA PREPARATION PIPELINE
Xử lý dữ liệu từ raw → processed (ready for training)
Bao gồm: Cleaning, Split, Imputation, Encoding, Scaling, Feature Engineering
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import config
import config

# Import preprocessing modules
from preprocessing.data_cleaning import DataCleaner
from preprocessing.imputation import Imputer
from preprocessing.encoding import Encoder
from preprocessing.scaling import Scaler
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.multi_label_encoding import MultiLabelEncoder

# Import modeling modules
from modeling.train_test_split import DataSplitter
from modeling.handle_imbalanced import ImbalanceHandler


class DataPreparationPipeline:
    """Class để chuẩn bị dữ liệu từ raw → processed"""
    
    def __init__(self, config_module=None):
        """
        Parameters:
        -----------
        config_module : module, optional
            Module config (default sử dụng config.py)
        """
        print("\n" + "="*70)
        print("1. XÓA DUPLICATES")
        print("="*70)
        self.config = config_module if config_module else config
        
        # Khởi tạo các preprocessors
        self.cleaner = None
        self.splitter = None
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.engineer = None
        self.imbalance_handler = None
        self.ml_encoder = None
        
        # Lưu trữ metadata
        self.metadata = {}
        
    def load_raw_data(self):
        """Load raw data"""
        df = pd.read_csv(self.config.RAW_DATA_PATH)
        
        self.metadata['raw_shape'] = df.shape
        
        return df
    
    def clean_data(self, df):
        """Step 1: Data Cleaning"""
        
        self.cleaner = DataCleaner(
            missing_threshold=self.config.MISSING_THRESHOLD,
            low_variance_threshold=self.config.LOW_VARIANCE_THRESHOLD,
            high_corr_threshold=self.config.HIGH_CORRELATION_THRESHOLD
        )
        
        df_cleaned = self.cleaner.clean(
            df,
            columns_to_drop=self.config.COLUMNS_TO_DROP,
            remove_high_corr=False
        )
        
        # Save cleaned data
        os.makedirs(os.path.dirname(self.config.CLEANED_DATA_PATH), exist_ok=True)
        df_cleaned.to_csv(self.config.CLEANED_DATA_PATH, index=False)
        print(f"\n✓ Saved cleaned data to {self.config.CLEANED_DATA_PATH}")
        
        self.metadata['cleaned_shape'] = df_cleaned.shape
        
        return df_cleaned
    
    def split_data(self, df):
        '""Step 2: Train/Val/Test Split""'
        
        self.splitter = DataSplitter(
            test_size=self.config.TEST_SIZE,
            validation_size=self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify_column=self.config.STRATIFY_COLUMN
        )
        
        if self.config.VALIDATION_SIZE > 0:
            X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split(
                df,
                target_col=self.config.TARGET_COLUMN,
                include_validation=True
            )
        else:
            X_train, X_test, y_train, y_test = self.splitter.split(
                df,
                target_col=self.config.TARGET_COLUMN,
                include_validation=False
            )
            X_val, y_val = None, None
        
        self.metadata['train_samples'] = len(X_train)
        self.metadata['val_samples'] = len(X_val) if X_val is not None else 0
        self.metadata['test_samples'] = len(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_multi_label(self, X_train, X_val, X_test):
        """Step 2.5: Multi-Label Encoding (Positions column)"""
        if not self.config.MULTI_LABEL_COLUMNS:
            return X_train, X_val, X_test
        
        # Check if columns exist
        cols_to_encode = [col for col in self.config.MULTI_LABEL_COLUMNS if col in X_train.columns]
        
        if not cols_to_encode:
            print("\n⚠️  No multi-label columns found, skipping...")
            return X_train, X_val, X_test
        
        print("\n" + "="*80)
        print("STEP 2.5: MULTI-LABEL ENCODING (Positions → Binary Features)")
        print("="*80)
        
        self.ml_encoder = MultiLabelEncoder(delimiter=self.config.MULTI_LABEL_DELIMITER)
        
        # Fit trên train
        self.ml_encoder.fit(X_train, cols_to_encode)
        
        # Transform tất cả
        X_train = self.ml_encoder.transform(X_train)
        X_test = self.ml_encoder.transform(X_test)
        if X_val is not None:
            X_val = self.ml_encoder.transform(X_val)
        
        print(f"\n✓ Multi-label encoding completed")
        print(f"  Created {len(self.ml_encoder.get_all_feature_names())} binary features")
        print(f"  Example: {', '.join(self.ml_encoder.get_all_feature_names()[:5])}...")
        
        return X_train, X_val, X_test
    
    def impute_missing_values(self, X_train, X_val, X_test):
        """Step 3: Imputation"""
        
        self.imputer = Imputer(
            numeric_strategy=self.config.NUMERIC_IMPUTATION_STRATEGY,
            categorical_strategy=self.config.CATEGORICAL_IMPUTATION_STRATEGY,
            knn_neighbors=self.config.KNN_NEIGHBORS
        )
        
        # Fit trên train, transform trên tất cả
        self.imputer.fit(X_train)
        X_train = self.imputer.transform(X_train)
        X_test = self.imputer.transform(X_test)
        if X_val is not None:
            X_val = self.imputer.transform(X_val)
        
        print(f"✓ Imputation completed")
        
        return X_train, X_val, X_test
    
    def encode_categorical(self, X_train, X_val, X_test, y_train):
        """Step 4: Encoding"""
        
        self.encoder = Encoder(
            method=self.config.CATEGORICAL_ENCODING_METHOD,
            max_categories=self.config.MAX_CATEGORIES
        )
        
        # Fit trên train, transform trên tất cả
        target = y_train if self.config.CATEGORICAL_ENCODING_METHOD == 'target' else None
        self.encoder.fit(X_train, target=target)
        X_train = self.encoder.transform(X_train)
        X_test = self.encoder.transform(X_test)
        if X_val is not None:
            X_val = self.encoder.transform(X_val)
        
        print(f"✓ Encoding completed")
        
        return X_train, X_val, X_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Step 5: Scaling"""
        
        self.scaler = Scaler(method=self.config.SCALING_METHOD)
        
        # Fit trên train, transform trên tất cả
        self.scaler.fit(X_train, exclude_cols=[])
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        
        print(f"✓ Scaling completed")
        
        return X_train, X_val, X_test
    
    def engineer_features(self, X_train, X_val, X_test, y_train):
        """Step 6: Feature Engineering"""
        
        self.engineer = FeatureEngineer(
            create_polynomial=self.config.CREATE_POLYNOMIAL_FEATURES,
            polynomial_degree=self.config.POLYNOMIAL_DEGREE,
            selection_method=self.config.FEATURE_SELECTION_METHOD,
            n_features=self.config.N_FEATURES_TO_SELECT
        )
        
        # Combine để engineer
        train_combined = X_train.copy()
        train_combined[self.config.TARGET_COLUMN] = y_train.values
        
        train_engineered = self.engineer.engineer(
            train_combined,
            target_col=self.config.TARGET_COLUMN,
            task_type=self.config.TASK_TYPE
        )
        
        X_train = train_engineered.drop(columns=[self.config.TARGET_COLUMN])
        y_train = train_engineered[self.config.TARGET_COLUMN]
        
        # Align features cho val và test
        selected_features = [col for col in X_train.columns if col in X_test.columns]
        X_test = X_test[selected_features]
        if X_val is not None:
            X_val = X_val[selected_features]
        
        print(f"✓ Feature engineering completed")
        print(f"  Final features: {len(selected_features)}")
        
        self.metadata['n_features'] = len(selected_features)
        self.metadata['feature_names'] = selected_features
        
        return X_train, X_val, X_test, y_train
    
    def handle_imbalanced(self, X_train, y_train):
        """Step 7: Handle Imbalanced (cho classification)"""
        if self.config.TASK_TYPE == 'classification' and self.config.HANDLE_IMBALANCED:
            
            try:
                self.imbalance_handler = ImbalanceHandler(
                    method=self.config.IMBALANCED_METHOD,
                    random_state=self.config.RANDOM_STATE
                )
                
                X_train, y_train = self.imbalance_handler.resample(X_train, y_train)
                print(f"✓ Resampling completed")
                self.metadata['resampled'] = True
            except Exception as e:
                print(f"⚠️  Resampling failed: {e}")
                self.metadata['resampled'] = False
        
        return X_train, y_train
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data"""
        
        os.makedirs('data/processed', exist_ok=True)
        
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
        
        if X_val is not None:
            X_val.to_csv('data/processed/X_val.csv', index=False)
            pd.DataFrame(y_val).to_csv('data/processed/y_val.csv', index=False)
        
        print(f"✓ Saved processed data to data/processed/")
    
    def save_preprocessors(self, X_train):
        """Save preprocessors để dùng cho inference"""
        
        os.makedirs('models/preprocessors', exist_ok=True)
        
        # Save từng preprocessor
        if self.imputer:
            with open('models/preprocessors/imputer.pkl', 'wb') as f:
                pickle.dump(self.imputer, f)
            print("✓ Saved imputer")
        
        if self.encoder:
            with open('models/preprocessors/encoder.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)
            print("✓ Saved encoder")
        
        if self.scaler:
            with open('models/preprocessors/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print("✓ Saved scaler")
        
        if self.engineer:
            with open('models/preprocessors/engineer.pkl', 'wb') as f:
                pickle.dump(self.engineer, f)
            print("✓ Saved engineer")
        
        # Save multi-label encoder
        if self.ml_encoder:
            with open('models/preprocessors/ml_encoder.pkl', 'wb') as f:
                pickle.dump(self.ml_encoder, f)
            print("✓ Saved multi-label encoder")
        
        # Save feature names
        with open('models/preprocessors/feature_names.pkl', 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
        print("✓ Saved feature names")
        
        # Save metadata
        with open('models/preprocessors/preparation_metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
        print("✓ Saved metadata")
    
    def run_full_pipeline(self):
        """Chạy toàn bộ data preparation pipeline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.metadata['timestamp'] = timestamp
        self.metadata['task_type'] = self.config.TASK_TYPE
        self.metadata['target_column'] = self.config.TARGET_COLUMN
        
        # Step 1: Load raw data
        df = self.load_raw_data()
        
        # Step 2: Clean data
        df_cleaned = self.clean_data(df)
        
        # Step 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df_cleaned)
        
        # Step 3.5: Multi-label encoding (Positions)
        X_train, X_val, X_test = self.encode_multi_label(X_train, X_val, X_test)
        
        # Step 4: Imputation
        X_train, X_val, X_test = self.impute_missing_values(X_train, X_val, X_test)
        
        # Step 5: Encoding
        X_train, X_val, X_test = self.encode_categorical(X_train, X_val, X_test, y_train)
        
        # Step 6: Scaling
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)
        
        # Step 7: Feature Engineering
        X_train, X_val, X_test, y_train = self.engineer_features(X_train, X_val, X_test, y_train)
        
        # Step 8: Handle Imbalanced (nếu cần)
        X_train, y_train = self.handle_imbalanced(X_train, y_train)
        
        # Save processed data
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Save preprocessors
        self.save_preprocessors(X_train)
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data():
    """Function wrapper để chạy pipeline"""
    pipeline = DataPreparationPipeline()
    return pipeline.run_full_pipeline()


if __name__ == "__main__":
    # Chạy data preparation pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    print(f"\n✓ Data is ready for training!")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    if X_val is not None:
        print(f"  - X_val shape: {X_val.shape}")
