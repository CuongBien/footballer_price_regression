"""
Module 8: Model Training
- Train nhiều models
- Hyperparameter tuning
- Cross-validation
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import config để lấy KNN_CONFIG
try:
    import config as cfg
except ImportError:
    cfg = None

# Import custom Decision Tree models
try:
    # Thêm path để import custom models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from decisionTree.Regression.regressionTree import RegressionTree as CustomRegressionTree
    from decisionTree.Classification.classificationTree import DecisionTreeClassifier as CustomDecisionTreeClassifier
    CUSTOM_TREE_AVAILABLE = True
except ImportError as e:
    CUSTOM_TREE_AVAILABLE = False
    print(f"Warning: Custom Decision Tree not available: {e}")

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import custom model
from modeling.hist_gradient_boosting import HistGradientBoostingModel


class ModelTrainer:
    """Class để train nhiều models"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42):
        """
        Parameters:
        -----------
        task_type : str
            Loại task: 'regression' hoặc 'classification'
        random_state : int
            Random seed
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.training_scores = {}
        
        # Khởi tạo models
        self._initialize_models()
    
    def _initialize_models(self):
        """Khởi tạo các models"""
        if self.task_type == 'regression':
            # Lấy KNN config từ config.py nếu có sẵn, nếu không dùng mặc định
            knn_config = {
                'n_neighbors': 30,
                'weights': 'uniform',
                'metric': 'euclidean'
            }
            if cfg:
                knn_config = cfg.KNN_CONFIG
            
            self.models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=self.random_state),
                'KNN': KNeighborsRegressor(
                    n_neighbors=knn_config['n_neighbors'],
                    weights=knn_config['weights'],
                    metric=knn_config['metric']
                ),
                'HistGradientBoosting': HistGradientBoostingModel(random_state=self.random_state),

                'DecisionTreeRegressor': DecisionTreeRegressor(
                    max_depth=10, min_samples_split=5, random_state=self.random_state
                ),
                'RandomForestRegressor': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
                ),

            }
            # Thêm custom RegressionTree nếu available
            if CUSTOM_TREE_AVAILABLE:
                self.models['CustomRegressionTree_MSE'] = CustomRegressionTree(
                    criterion='mse', max_depth=10, min_samples_split=5
                )
                self.models['CustomRegressionTree_MAE'] = CustomRegressionTree(
                    criterion='mae', max_depth=10, min_samples_split=5
                )
        else:  # classification
            self.models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(
                    max_depth=10, min_samples_split=5, random_state=self.random_state
                ),
                'RandomForestClassifier': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
                ),
            }
            # Thêm custom DecisionTreeClassifier nếu available
            if CUSTOM_TREE_AVAILABLE:
                self.models['CustomDecisionTree_IG'] = CustomDecisionTreeClassifier(
                    criterion='information_gain', max_depth=10, min_samples_split=5
                )
                self.models['CustomDecisionTree_Gini'] = CustomDecisionTreeClassifier(
                    criterion='gini', max_depth=10, min_samples_split=5
                )
            if XGBOOST_AVAILABLE:
                self.models['XGBoost'] = XGBClassifier(random_state=self.random_state, n_jobs=-1)
    
    def add_model(self, name: str, model: Any):
        """
        Thêm model mới vào danh sách models
        
        Parameters:
        -----------
        name : str
            Tên model
        model : Any
            Model instance (phải có phương thức fit, predict, score)
        """
        self.models[name] = model
        print(f"✓ Added model: {name}")
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None) -> Any:
        """
        Train một model đơn lẻ
        
        Parameters:
        -----------
        model_name : str
            Tên model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
            
        Returns:
        --------
        Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' không tồn tại")
        
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Đánh giá trên train set
        train_score = model.score(X_train, y_train)
        print(f"  - Train score: {train_score:.4f}")
        
        # Đánh giá trên validation set (nếu có)
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            print(f"  - Validation score: {val_score:.4f}")
            self.training_scores[model_name] = {
                'train': train_score,
                'validation': val_score
            }
        else:
            self.training_scores[model_name] = {
                'train': train_score
            }
        
        # Lưu trained model
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train tất cả models
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        models_to_train : List[str], optional
            Danh sách models muốn train (None = train tất cả)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary của trained models
        """
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        print(f"\nTask type: {self.task_type.upper()}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Training samples: {len(X_train)}")
        
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        
        # Xác định models cần train
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        print(f"\nTraining {len(models_to_train)} models...")
        
        # Train từng model
        for model_name in models_to_train:
            try:
                self.train_single_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
        
        # Tóm tắt kết quả
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        if self.training_scores:
            scores_df = pd.DataFrame(self.training_scores).T
            # Sort by first available column
            sort_col = 'validation' if 'validation' in scores_df.columns else 'train'
            scores_df = scores_df.sort_values(sort_col, ascending=False)
            print("\n" + str(scores_df))
        else:
            print("\nNo models were trained successfully.")
        
        return self.trained_models
    
    def cross_validate_model(self,
                            model_name: str,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv: int = 5) -> Dict:
        """
        Cross-validation cho một model
        
        Parameters:
        -----------
        model_name : str
            Tên model
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Số folds
            
        Returns:
        --------
        Dict
            Kết quả cross-validation
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' không tồn tại")
        
        print(f"\nCross-validating {model_name} ({cv}-fold)...")
        
        model = self.models[model_name]
        
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
        
        result = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"  - Mean score: {result['mean']:.4f} (+/- {result['std']:.4f})")
        
        return result
    
    def save_models(self, save_dir: str = '../models/'):
        """
        Lưu tất cả trained models
        
        Parameters:
        -----------
        save_dir : str
            Thư mục để lưu models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving {len(self.trained_models)} models to {save_dir}...")
        
        for model_name, model in self.trained_models.items():
            filepath = os.path.join(save_dir, f"{model_name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Saved {model_name}")
        
        # Lưu training scores
        scores_filepath = os.path.join(save_dir, "training_scores.pkl")
        with open(scores_filepath, 'wb') as f:
            pickle.dump(self.training_scores, f)
        
        print(f"\n✓ All models saved to {save_dir}")
    
    def load_model(self, model_name: str, load_dir: str = '../models/') -> Any:
        """
        Load một trained model
        
        Parameters:
        -----------
        model_name : str
            Tên model
        load_dir : str
            Thư mục chứa models
            
        Returns:
        --------
        Loaded model
        """
        import os
        filepath = os.path.join(load_dir, f"{model_name}.pkl")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Loaded {model_name} from {filepath}")
        
        return model
    
    def get_best_model(self, metric: str = 'validation') -> tuple:
        """
        Lấy model tốt nhất
        
        Parameters:
        -----------
        metric : str
            Metric để chọn: 'train' hoặc 'validation'
            
        Returns:
        --------
        tuple : (model_name, model, score)
        """
        if not self.training_scores:
            raise ValueError("Chưa có model nào được train")
        
        best_model_name = None
        best_score = -np.inf
        
        for model_name, scores in self.training_scores.items():
            if metric in scores and scores[metric] > best_score:
                best_score = scores[metric]
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Không tìm thấy metric '{metric}'")
        
        best_model = self.trained_models[best_model_name]
        
        print(f"\n✓ Best model: {best_model_name}")
        print(f"  {metric} score: {best_score:.4f}")
        
        return best_model_name, best_model, best_score


if __name__ == "__main__":
    # Test
    X_train = pd.read_csv('../data/X_train.csv')
    X_val = pd.read_csv('../data/X_val.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze()
    y_val = pd.read_csv('../data/y_val.csv').squeeze()
    
    trainer = ModelTrainer(task_type='regression', random_state=42)
    
    trained_models = trainer.train_all_models(
        X_train, y_train, X_val, y_val
    )
    
    # Get best model
    best_name, best_model, best_score = trainer.get_best_model(metric='validation')
    
    # Save models
    trainer.save_models('../models/')

    # Save models
    trainer.save_models('../models/')
