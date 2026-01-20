"""
EVALUATE PIPELINE
Đánh giá models đã train trên nhiều test sets khác nhau
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluatorAdvanced:
    """Class để đánh giá chi tiết models đã train"""
    
    def __init__(self, model_path: str = 'models/', results_path: str = 'results/'):
        self.model_path = model_path
        self.results_path = results_path
        self.models = {}
        self.metadata = {}
        
        # Load models và metadata
        self._load_all_models()
        self._load_metadata()
    
    def _load_all_models(self):
        """Load tất cả trained models"""
        print("="*70)
        print("LOADING ALL MODELS")
        print("="*70)
        
        # Check for log transformer
        log_transformer_path = os.path.join(self.model_path, 'preprocessors', 'log_transformer.pkl')
        if os.path.exists(log_transformer_path):
            from preprocessing.log_transform import LogTargetTransformer
            self.log_transformer = LogTargetTransformer.load(log_transformer_path)
            print("✓ Loaded log transformer")
        else:
            self.log_transformer = None
        
        for filename in os.listdir(self.model_path):
            if filename.endswith('.pkl') and filename not in ['training_metadata.pkl', 'training_scores.pkl']:
                model_name = filename.replace('.pkl', '')
                filepath = os.path.join(self.model_path, filename)
                
                with open(filepath, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                print(f"✓ Loaded {model_name}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
    
    def _load_metadata(self):
        """Load training metadata"""
        metadata_path = os.path.join(self.model_path, 'training_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
    
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Đánh giá tất cả models trên test set
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        pd.DataFrame
            Bảng kết quả
        """
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS ON TEST SET")
        print("="*70)
        
        results = []
        task_type = self.metadata.get('task_type', 'regression')
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                y_pred = model.predict(X_test)
                
                # Inverse transform if log transformer exists
                if self.log_transformer is not None and task_type == 'regression':
                    y_pred = self.log_transformer.inverse_transform(y_pred)
                
                if task_type == 'regression':
                    metrics = {
                        'Model': model_name,
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'R2': r2_score(y_test, y_pred)
                    }
                    
                    try:
                        metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
                    except:
                        metrics['MAPE'] = None
                    
                else:  # classification
                    metrics = {
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                    
                    # ROC-AUC nếu có predict_proba
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)
                            if len(np.unique(y_test)) == 2:
                                metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba[:, 1])
                            else:
                                metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        except:
                            metrics['ROC-AUC'] = None
                
                results.append(metrics)
                
                # Print metrics
                for metric, value in metrics.items():
                    if metric != 'Model' and value is not None:
                        print(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        results_df = pd.DataFrame(results)
        
        # Sort by main metric
        if task_type == 'regression':
            results_df = results_df.sort_values('R2', ascending=False)
        else:
            results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """Vẽ biểu đồ so sánh các models"""
        print("\n" + "="*70)
        print("PLOTTING MODEL COMPARISON")
        print("="*70)
        
        task_type = self.metadata.get('task_type', 'regression')
        
        if task_type == 'regression':
            metrics = ['MAE', 'RMSE', 'R2']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                figsize=(6*len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            data = results_df.sort_values(metric, ascending=(metric in ['MAE', 'RMSE']))
            
            axes[idx].barh(range(len(data)), data[metric].values, color='skyblue')
            axes[idx].set_yticks(range(len(data)))
            axes[idx].set_yticklabels(data['Model'].values)
            axes[idx].set_xlabel(metric, fontsize=12)
            axes[idx].set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(data[metric].values):
                axes[idx].text(v, i, f' {v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_path, 'model_comparison_detailed.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
        plt.close()
    
    def plot_regression_analysis(self, 
                                 model_name: str,
                                 X_test: pd.DataFrame, 
                                 y_test: pd.Series,
                                 save_path: str = None):
        """Vẽ biểu đồ phân tích chi tiết cho regression"""
        print(f"\nPlotting regression analysis for {model_name}...")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=30)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values', fontsize=12)
        axes[0, 0].set_ylabel('Predicted Values', fontsize=12)
        axes[0, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add metrics text
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fig.suptitle(f'{model_name} - Regression Analysis\n' + 
                    f'MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_path, f'{model_name}_regression_analysis.png')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.close()
    
    def generate_full_report(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Tạo báo cáo đầy đủ về tất cả models"""
        print("\n" + "="*80)
        print(" " * 20 + "GENERATING FULL EVALUATION REPORT")
        print("="*80)
        
        # Evaluate all models
        results_df = self.evaluate_on_test_set(X_test, y_test)
        
        # Save results
        os.makedirs(self.results_path, exist_ok=True)
        results_path = os.path.join(self.results_path, 'evaluation_report.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Saved evaluation report to {results_path}")
        
        # Plot comparison
        self.plot_comparison(results_df)
        
        # Plot detailed analysis for best model
        best_model_name = results_df.iloc[0]['Model']
        
        if self.metadata.get('task_type') == 'regression':
            self.plot_regression_analysis(best_model_name, X_test, y_test)
        
        print("\n" + "="*80)
        print("FULL REPORT GENERATED!")
        print("="*80)
        print(f"\nFiles created:")
        print(f"  - {results_path}")
        print(f"  - {self.results_path}/model_comparison_detailed.png")
        if self.metadata.get('task_type') == 'regression':
            print(f"  - {self.results_path}/{best_model_name}_regression_analysis.png")


def evaluate_pipeline():
    """Main evaluation pipeline"""
    
    print("="*80)
    print(" " * 25 + "EVALUATION PIPELINE")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluatorAdvanced(model_path='models/', results_path='results/')
    
    # Generate full report
    evaluator.generate_full_report(X_test, y_test)
    
    print("\n✓ Evaluation pipeline completed!")


if __name__ == "__main__":
    evaluate_pipeline()
