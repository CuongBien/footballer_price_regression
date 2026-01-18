"""
Module 8: Model Evaluation
- Đánh giá models trên test set
- Tính toán metrics
- Vẽ biểu đồ kết quả
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    # Regression metrics
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class để đánh giá models"""
    
    def __init__(self, task_type: str = 'regression'):
        """
        Parameters:
        -----------
        task_type : str
            Loại task: 'regression' hoặc 'classification'
        """
        self.task_type = task_type
        self.evaluation_results = {}
        
    def evaluate_regression(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = 'model') -> Dict:
        """
        Đánh giá regression model
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Tên model
            
        Returns:
        --------
        Dict
            Dictionary chứa các metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (nếu không có giá trị 0)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = None
        
        results = {
            'model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_classification(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               model_name: str = 'model') -> Dict:
        """
        Đánh giá classification model
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities (cho ROC-AUC)
        model_name : str
            Tên model
            
        Returns:
        --------
        Dict
            Dictionary chứa các metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        # Xử lý binary và multiclass
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC (nếu có probabilities)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        results = {
            'model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_model(self,
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str = 'model') -> Dict:
        """
        Đánh giá một model trên test set
        
        Parameters:
        -----------
        model : trained model
            Model đã được train
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        model_name : str
            Tên model
            
        Returns:
        --------
        Dict
            Kết quả đánh giá
        """
        print(f"\nEvaluating {model_name}...")
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Đánh giá theo task type
        if self.task_type == 'regression':
            results = self.evaluate_regression(y_test, y_pred, model_name)
            
            print(f"  MAE: {results['MAE']:.4f}")
            print(f"  RMSE: {results['RMSE']:.4f}")
            print(f"  R2: {results['R2']:.4f}")
            
        else:  # classification
            # Lấy probabilities nếu có
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            results = self.evaluate_classification(
                y_test, 
                y_pred, 
                y_pred_proba, 
                model_name
            )
            
            print(f"  Accuracy: {results['Accuracy']:.4f}")
            print(f"  F1-Score: {results['F1-Score']:.4f}")
            if results['ROC-AUC']:
                print(f"  ROC-AUC: {results['ROC-AUC']:.4f}")
        
        return results
    
    def evaluate_multiple_models(self,
                                models: Dict[str, Any],
                                X_test: pd.DataFrame,
                                y_test: pd.Series) -> pd.DataFrame:
        """
        Đánh giá nhiều models
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary của trained models
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        pd.DataFrame
            Bảng so sánh kết quả
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION ON TEST SET")
        print("="*70)
        
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Tạo bảng kết quả
        results_df = pd.DataFrame(self.evaluation_results).T
        
        # Sort theo metric chính
        if self.task_type == 'regression':
            results_df = results_df.sort_values('R2', ascending=False)
        else:
            results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print("\n" + str(results_df))
        
        return results_df
    
    def plot_regression_results(self,
                               model: Any,
                               X_test: pd.DataFrame,
                               y_test: pd.Series,
                               model_name: str = 'model',
                               save_path: Optional[str] = None):
        """
        Vẽ biểu đồ kết quả regression
        
        Parameters:
        -----------
        model : trained model
            Model đã được train
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        model_name : str
            Tên model
        save_path : str, optional
            Đường dẫn để lưu hình
        """
        y_pred = model.predict(X_test)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'{model_name}: Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
    
    def plot_classification_results(self,
                                   model: Any,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   model_name: str = 'model',
                                   save_path: Optional[str] = None):
        """
        Vẽ biểu đồ kết quả classification
        
        Parameters:
        -----------
        model : trained model
            Model đã được train
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        model_name : str
            Tên model
        save_path : str, optional
            Đường dẫn để lưu hình
        """
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name}: Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.show()
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def plot_comparison(self, 
                       results_df: pd.DataFrame,
                       save_path: Optional[str] = None):
        """
        Vẽ biểu đồ so sánh các models
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Bảng kết quả từ evaluate_multiple_models
        save_path : str, optional
            Đường dẫn để lưu hình
        """
        if self.task_type == 'regression':
            metrics = ['MAE', 'RMSE', 'R2']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Lọc metrics có giá trị
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("Không có metrics để vẽ")
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                figsize=(6*len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            data = results_df[metric].sort_values(ascending=(metric in ['MAE', 'MSE', 'RMSE']))
            
            axes[idx].barh(range(len(data)), data.values)
            axes[idx].set_yticks(range(len(data)))
            axes[idx].set_yticklabels(data.index)
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'Comparison: {metric}')
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        plt.show()
    
    def get_evaluation_report(self) -> Dict:
        """Lấy báo cáo đánh giá chi tiết"""
        return self.evaluation_results


if __name__ == "__main__":
    # Test
    import pickle
    
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv').squeeze()
    
    # Load trained models
    models = {}
    import os
    model_dir = '../models/'
    
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl') and filename != 'training_scores.pkl':
            model_name = filename.replace('.pkl', '')
            with open(os.path.join(model_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Evaluate
    evaluator = ModelEvaluator(task_type='regression')
    results_df = evaluator.evaluate_multiple_models(models, X_test, y_test)
    
    # Save results
    results_df.to_csv('../results/evaluation_results.csv')
    print("\n✓ Saved results to results/evaluation_results.csv")
    
    # Plot comparison
    evaluator.plot_comparison(results_df, save_path='../results/model_comparison.png')
