"""
TRAINING PIPELINE
Chỉ dùng để train models (data đã được chuẩn bị từ data_preparation_pipeline.py)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import config
import config

# Import modeling modules
from modeling.model_training import ModelTrainer
from modeling.model_evaluation import ModelEvaluator

import os
import pickle
from datetime import datetime


def train_pipeline(use_prepared_data=True):
    """
    Pipeline để train models
    
    Parameters:
    -----------
    use_prepared_data : bool
        True: Load dữ liệu đã chuẩn bị từ data/processed/
        False: Chạy data preparation từ đầu
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print(" " * 20 + "TRAINING PIPELINE - FOOTBALLER")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    
    # ========================================
    # LOAD DATA
    # ========================================
    if use_prepared_data:
        print("\n" + "="*80)
        print("LOADING PREPARED DATA")
        print("="*80)
        
        # Check nếu data đã được chuẩn bị
        if not os.path.exists('data/processed/X_train.csv'):
            print("[WARNING] Processed data does not exist!")
            print("Chạy data_preparation_pipeline.py trước hoặc set use_prepared_data=False")
            print("\nĐang chạy data preparation pipeline...")
            
            from data_preparation_pipeline import prepare_data
            X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
        else:
            # Load processed data
            X_train = pd.read_csv('data/processed/X_train.csv')
            X_test = pd.read_csv('data/processed/X_test.csv')
            y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
            y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
            
            if os.path.exists('data/processed/X_val.csv'):
                X_val = pd.read_csv('data/processed/X_val.csv')
                y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
            else:
                X_val, y_val = None, None
            
            print(f"[OK] Loaded processed data")
            print(f"  - Train: {X_train.shape}")
            print(f"  - Test: {X_test.shape}")
            if X_val is not None:
                print(f"  - Validation: {X_val.shape}")
    else:
        print("\n" + "="*80)
        print("RUNNING DATA PREPARATION PIPELINE")
        print("="*80)
        
        from data_preparation_pipeline import prepare_data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    # ========================================
    # MODEL TRAINING
    # ========================================
    # LOG TRANSFORM TARGET (if enabled)
    # ========================================
    if config.TASK_TYPE == 'regression' and hasattr(config, 'USE_LOG_TRANSFORM') and config.USE_LOG_TRANSFORM:
        print("\n" + "="*80)
        print("LOG TRANSFORM TARGET")
        print("="*80)
        
        from preprocessing.log_transform import LogTargetTransformer
        
        log_transformer = LogTargetTransformer()
        
        print(f"\nOriginal target range:")
        print(f"  Min:  €{y_train.min():,.0f}")
        print(f"  Max:  €{y_train.max():,.0f}")
        print(f"  Mean: €{y_train.mean():,.0f}")
        
        # Transform
        y_train_transformed = log_transformer.fit_transform(y_train)
        y_val_transformed = log_transformer.transform(y_val) if y_val is not None else None
        y_test_transformed = log_transformer.transform(y_test)
        
        print(f"\nLog-transformed target range:")
        print(f"  Min:  {y_train_transformed.min():.4f}")
        print(f"  Max:  {y_train_transformed.max():.4f}")
        print(f"  Mean: {y_train_transformed.mean():.4f}")
        
        # Save transformer
        log_transformer.save(os.path.join(config.MODEL_SAVE_PATH, 'preprocessors', 'log_transformer.pkl'))
        print("✓ Saved log transformer")
        
        # Use transformed targets for training
        y_train_for_training = y_train_transformed
        y_val_for_training = y_val_transformed
    else:
        y_train_for_training = y_train
        y_val_for_training = y_val
        log_transformer = None
    
    # ========================================
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    trainer = ModelTrainer(
        task_type=config.TASK_TYPE,
        random_state=config.RANDOM_STATE
    )
    
    models_to_train = config.MODELS.get(config.TASK_TYPE, None)
    
    trained_models = trainer.train_all_models(
        X_train, y_train_for_training,
        X_val, y_val_for_training,
        models_to_train=models_to_train
    )
    
    best_name, best_model, best_score = trainer.get_best_model(
        metric='validation' if X_val is not None else 'train'
    )
    
    # Save models
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    trainer.save_models(config.MODEL_SAVE_PATH)
    
    # ========================================
    # MODEL EVALUATION
    # ========================================
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    evaluator = ModelEvaluator(task_type=config.TASK_TYPE)
    
    # Evaluate trên validation set (nếu có)
    if X_val is not None:
        print("\n--- VALIDATION SET ---")
        val_results = evaluator.evaluate_multiple_models(trained_models, X_val, y_val)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        val_results.to_csv(f'{config.RESULTS_PATH}/validation_results.csv')
    
    # Evaluate trên test set
    print("\n--- TEST SET ---")
    test_results = evaluator.evaluate_multiple_models(trained_models, X_test, y_test)
    
    # Save results
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    test_results.to_csv(f'{config.RESULTS_PATH}/test_results.csv')
    
    # Plot comparison
    if config.SAVE_PLOTS:
        evaluator.plot_comparison(
            test_results,
            save_path=f'{config.RESULTS_PATH}/model_comparison.{config.PLOT_FORMAT}'
        )
        
        # Plot best model results
        if config.TASK_TYPE == 'regression':
            evaluator.plot_regression_results(
                best_model, X_test, y_test,
                model_name=best_name,
                save_path=f'{config.RESULTS_PATH}/{best_name}_results.{config.PLOT_FORMAT}'
            )
        else:
            evaluator.plot_classification_results(
                best_model, X_test, y_test,
                model_name=best_name,
                save_path=f'{config.RESULTS_PATH}/{best_name}_confusion_matrix.{config.PLOT_FORMAT}'
            )
    
    # ========================================
    # SAVE TRAINING METADATA
    # ========================================
    metadata = {
        'timestamp': timestamp,
        'train_samples': len(X_train),
        'val_samples': len(X_val) if X_val is not None else 0,
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'target_column': config.TARGET_COLUMN,
        'task_type': config.TASK_TYPE,
        'best_model': best_name,
        'best_score': float(best_score),
        'models_trained': list(trained_models.keys()),
        'test_results': test_results.to_dict()
    }
    
    with open(f'{config.MODEL_SAVE_PATH}/training_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n" + "="*80)
    print(" " * 25 + "TRAINING COMPLETED!")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"  - Final features: {X_train.shape[1]}")
    print(f"  - Train samples: {len(X_train):,}")
    if X_val is not None:
        print(f"  - Validation samples: {len(X_val):,}")
    print(f"  - Test samples: {len(X_test):,}")
    print(f"  - Models trained: {len(trained_models)}")
    print(f"  - Best model: {best_name} (score: {best_score:.4f})")
    
    print(f"\n📁 OUTPUT:")
    print(f"  - Trained models: {config.MODEL_SAVE_PATH}")
    print(f"  - Results: {config.RESULTS_PATH}")
    
    print("\n[OK] Training pipeline completed!")
    print(f"[OK] Use inference_pipeline.py to test on new data")
    
    return metadata


if __name__ == "__main__":
    metadata = train_pipeline()
