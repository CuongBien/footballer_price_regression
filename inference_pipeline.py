"""
INFERENCE PIPELINE
Dùng models đã train để predict trên dữ liệu mới
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Union, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ModelInference:
    """Class để load model và predict"""
    
    def __init__(self, model_path: str = 'models/', model_name: str = None):
        """
        Parameters:
        -----------
        model_path : str
            Đường dẫn đến thư mục chứa models
        model_name : str, optional
            Tên model cụ thể (nếu None sẽ load best model)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.preprocessors = {}
        self.feature_names = []
        self.metadata = {}
        
        # Load model và preprocessors
        self._load_model()
        self._load_preprocessors()
        self._load_metadata()
    
    def _load_model(self):
        """Load trained model"""
        print("="*70)
        print("LOADING MODEL")
        print("="*70)
        
        # Nếu không chỉ định model, load best model từ metadata
        if self.model_name is None:
            metadata_path = os.path.join(self.model_path, 'training_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.model_name = metadata.get('best_model', 'random_forest')
                print(f"Sử dụng best model: {self.model_name}")
            else:
                self.model_name = 'random_forest'
                print(f"Không tìm thấy metadata, sử dụng: {self.model_name}")
        
        # Load model
        model_file = os.path.join(self.model_path, f'{self.model_name}.pkl')
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file không tồn tại: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Loaded model: {self.model_name}")
    
    def _load_preprocessors(self):
        """Load các preprocessors (imputer, encoder, scaler, engineer)"""
        print("\nLOADING PREPROCESSORS")
        print("-"*70)
        
        preprocessor_path = os.path.join(self.model_path, 'preprocessors')
        
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessors không tồn tại: {preprocessor_path}")
        
        # Load từng preprocessor
        preprocessor_files = {
            'imputer': 'imputer.pkl',
            'encoder': 'encoder.pkl',
            'scaler': 'scaler.pkl',
            'engineer': 'engineer.pkl'
        }
        
        for name, filename in preprocessor_files.items():
            filepath = os.path.join(preprocessor_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.preprocessors[name] = pickle.load(f)
                print(f"✓ Loaded {name}")
            else:
                print(f"⚠️  {name} không tồn tại, bỏ qua")
        
        # Load multi-label encoder
        ml_encoder_path = os.path.join(preprocessor_path, 'ml_encoder.pkl')
        if os.path.exists(ml_encoder_path):
            with open(ml_encoder_path, 'rb') as f:
                self.preprocessors['ml_encoder'] = pickle.load(f)
            print(f"✓ Loaded multi-label encoder")
        
        # Load feature names
        feature_names_path = os.path.join(preprocessor_path, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✓ Loaded {len(self.feature_names)} feature names")
    
    def _load_metadata(self):
        """Load training metadata"""
        metadata_path = os.path.join(self.model_path, 'training_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"\n✓ Loaded training metadata")
            print(f"  - Trained on: {self.metadata.get('timestamp', 'Unknown')}")
            print(f"  - Task type: {self.metadata.get('task_type', 'Unknown')}")
            print(f"  - Best score: {self.metadata.get('best_score', 'N/A')}")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng các bước preprocessing lên dữ liệu mới
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dữ liệu cần xử lý
            
        Returns:
        --------
        pd.DataFrame
            Dữ liệu đã được xử lý
        """
        print("\n" + "="*70)
        print("PREPROCESSING NEW DATA")
        print("="*70)
        
        df_processed = df.copy()
        
        print(f"Input shape: {df_processed.shape}")
        
        # 0.5. Multi-label encoding
        if 'ml_encoder' in self.preprocessors:
            print("\n0.5. Applying multi-label encoding...")
            df_processed = self.preprocessors['ml_encoder'].transform(df_processed)
            print(f"   ✓ Shape: {df_processed.shape}")
        
        # 1. Imputation
        if 'imputer' in self.preprocessors:
            print("\n1. Applying imputation...")
            df_processed = self.preprocessors['imputer'].transform(df_processed)
            print(f"   ✓ Shape: {df_processed.shape}")
        
        # 2. Encoding
        if 'encoder' in self.preprocessors:
            print("\n2. Applying encoding...")
            df_processed = self.preprocessors['encoder'].transform(df_processed)
            print(f"   ✓ Shape: {df_processed.shape}")
        
        # 3. Scaling
        if 'scaler' in self.preprocessors:
            print("\n3. Applying scaling...")
            df_processed = self.preprocessors['scaler'].transform(df_processed)
            print(f"   ✓ Shape: {df_processed.shape}")
        
        # 4. Đảm bảo có đúng features (theo thứ tự)
        if self.feature_names:
            print("\n4. Aligning features...")
            
            # Thêm các features thiếu (fill với 0)
            for feature in self.feature_names:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
                    print(f"   + Added missing feature: {feature}")
            
            # Xóa các features thừa
            extra_features = set(df_processed.columns) - set(self.feature_names)
            if extra_features:
                df_processed = df_processed.drop(columns=list(extra_features))
                print(f"   - Removed {len(extra_features)} extra features")
            
            # Sắp xếp theo thứ tự
            df_processed = df_processed[self.feature_names]
            print(f"   ✓ Final shape: {df_processed.shape}")
        
        print("\n✓ Preprocessing completed")
        
        return df_processed
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán trên dữ liệu mới
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dữ liệu cần dự đoán
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        print("\n" + "="*70)
        print("MAKING PREDICTIONS")
        print("="*70)
        
        # Preprocess
        df_processed = self.preprocess(df)
        
        # Predict
        print(f"\nPredicting với {self.model_name}...")
        predictions = self.model.predict(df_processed)
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán probabilities (chỉ cho classification)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dữ liệu cần dự đoán
            
        Returns:
        --------
        np.ndarray
            Prediction probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Model {self.model_name} không hỗ trợ predict_proba")
        
        # Preprocess
        df_processed = self.preprocess(df)
        
        # Predict probabilities
        print(f"\nPredicting probabilities với {self.model_name}...")
        predictions = self.model.predict_proba(df_processed)
        
        print(f"✓ Generated {len(predictions)} probability predictions")
        
        return predictions
    
    def evaluate(self, df: pd.DataFrame, y_true: pd.Series) -> Dict:
        """
        Đánh giá model trên dữ liệu có ground truth
        
        Parameters:
        -----------
        df : pd.DataFrame
            Features
        y_true : pd.Series
            Ground truth labels
            
        Returns:
        --------
        Dict
            Evaluation metrics
        """
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        
        # Predict
        y_pred = self.predict(df)
        
        # Calculate metrics based on task type
        task_type = self.metadata.get('task_type', 'regression')
        
        if task_type == 'regression':
            metrics = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred)
            }
            
            print("\nRegression Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        else:  # classification
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            print("\nClassification Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin về model"""
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }


def inference_pipeline(input_data: Union[str, pd.DataFrame],
                      model_name: str = None,
                      model_path: str = 'models/',
                      output_file: str = None):
    """
    Pipeline inference đơn giản
    
    Parameters:
    -----------
    input_data : str or pd.DataFrame
        File path hoặc DataFrame
    model_name : str, optional
        Tên model cụ thể
    model_path : str
        Đường dẫn đến models
    output_file : str, optional
        File để lưu predictions
    """
    print("="*80)
    print(" " * 25 + "INFERENCE PIPELINE")
    print("="*80)
    
    # Load data
    if isinstance(input_data, str):
        print(f"\nLoading data from: {input_data}")
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    
    print(f"Input data shape: {df.shape}")
    
    # Initialize inference
    inference = ModelInference(model_path=model_path, model_name=model_name)
    
    # Make predictions
    predictions = inference.predict(df)
    
    # Add predictions to dataframe
    df['predictions'] = predictions
    
    # Save nếu có output_file
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved predictions to: {output_file}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED!")
    print("="*80)
    
    # Show sample predictions
    print("\nSample predictions:")
    print(df[['predictions']].head(10))
    
    print(f"\nPredictions statistics:")
    print(df['predictions'].describe())
    
    return df


if __name__ == "__main__":
    # Example 1: Predict trên test set
    print("\n" + "="*80)
    print("EXAMPLE 1: Predict trên test set")
    print("="*80)
    
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    inference = ModelInference(model_path='models/')
    
    # Evaluate
    metrics = inference.evaluate(X_test, y_test)
    
    # Get predictions
    predictions = inference.predict(X_test)
    
    # Save
    results_df = X_test.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = predictions
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/inference_results.csv', index=False)
    
    print("\n✓ Saved inference results to results/inference_results.csv")
    
    # Example 2: Predict trên dữ liệu mới (không có label)
    print("\n" + "="*80)
    print("EXAMPLE 2: Predict trên dữ liệu mới")
    print("="*80)
    
    # Giả sử bạn có file new_data.csv
    # df_new = inference_pipeline('new_data.csv', output_file='results/new_predictions.csv')
    print("Để predict trên dữ liệu mới:")
    print("  df_new = inference_pipeline('new_data.csv', output_file='results/new_predictions.csv')")
