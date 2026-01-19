"""
MAIN ML PIPELINE - FOOTBALLER RATING PREDICTION
Chạy toàn bộ pipeline: Data Preparation → Training → Evaluation
Lưu models vào folder models/ và báo cáo vào folder results/
"""

import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import config
from data_preparation_pipeline import DataPreparationPipeline
from train_pipeline import train_pipeline
from evaluate_pipeline import evaluate_pipeline


def create_folders():
    """Tạo các thư mục cần thiết"""
    folders = [
        config.MODEL_SAVE_PATH,           # models/
        config.RESULTS_PATH,              # results/
        'models/preprocessors',           # models/preprocessors/
        'data/processed',                 # data/processed/
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created/verified folder: {folder}")


def generate_report(prep_metadata, train_metadata):
    """
    Tạo báo cáo tổng hợp về quá trình training
    
    Parameters:
    -----------
    prep_metadata : dict
        Metadata từ data preparation
    train_metadata : dict
        Metadata từ training
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'timestamp': timestamp,
        'pipeline_info': {
            'task_type': config.TASK_TYPE,
            'target_column': config.TARGET_COLUMN,
            'random_state': config.RANDOM_STATE,
        },
        'data_preparation': {
            'raw_data_path': config.RAW_DATA_PATH,
            'missing_threshold': config.MISSING_THRESHOLD,
            'scaling_method': config.SCALING_METHOD,
            'encoding_method': config.CATEGORICAL_ENCODING_METHOD,
        },
        'training_results': {
            'train_samples': train_metadata.get('train_samples', 0),
            'val_samples': train_metadata.get('val_samples', 0),
            'test_samples': train_metadata.get('test_samples', 0),
            'n_features': train_metadata.get('n_features', 0),
            'models_trained': train_metadata.get('models_trained', []),
            'best_model': train_metadata.get('best_model', 'N/A'),
            'best_score': train_metadata.get('best_score', 0),
        },
        'test_results': train_metadata.get('test_results', {}),
    }
    
    # Lưu báo cáo JSON
    report_path = f"{config.RESULTS_PATH}/pipeline_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False, default=str)
    
    # Tạo báo cáo text đẹp
    text_report_path = f"{config.RESULTS_PATH}/pipeline_report_{timestamp}.txt"
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "ML PIPELINE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Task Type: {config.TASK_TYPE.upper()}\n")
        f.write(f"Target Column: {config.TARGET_COLUMN}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train samples: {train_metadata.get('train_samples', 0):,}\n")
        f.write(f"Validation samples: {train_metadata.get('val_samples', 0):,}\n")
        f.write(f"Test samples: {train_metadata.get('test_samples', 0):,}\n")
        f.write(f"Number of features: {train_metadata.get('n_features', 0)}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("MODELS TRAINED\n")
        f.write("-" * 40 + "\n")
        for model in train_metadata.get('models_trained', []):
            f.write(f"  • {model}\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("BEST MODEL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {train_metadata.get('best_model', 'N/A')}\n")
        f.write(f"Score: {train_metadata.get('best_score', 0):.4f}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("OUTPUT LOCATIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trained models: {config.MODEL_SAVE_PATH}\n")
        f.write(f"Results & reports: {config.RESULTS_PATH}\n")
        f.write(f"Processed data: data/processed/\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n📄 Report saved to:")
    print(f"   - {report_path}")
    print(f"   - {text_report_path}")
    
    return report


def main_pipeline():
    """
    Main pipeline thực hiện toàn bộ quy trình ML:
    1. Tạo các thư mục cần thiết
    2. Data Preparation (cleaning, encoding, scaling, split)
    3. Model Training (train all configured models)
    4. Model Evaluation (evaluate on test set)
    5. Tạo báo cáo tổng hợp
    
    Returns:
    --------
    dict
        Kết quả của toàn bộ pipeline
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "MAIN ML PIPELINE - FOOTBALLER PREDICTION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task type: {config.TASK_TYPE}")
    print(f"Target: {config.TARGET_COLUMN}")
    
    # Step 0: Tạo folders
    print("\n" + "=" * 80)
    print("STEP 0: CREATING FOLDERS")
    print("=" * 80)
    create_folders()
    
    # Step 1: Data Preparation
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)
    prep_pipeline = DataPreparationPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = prep_pipeline.run_full_pipeline()
    prep_metadata = getattr(prep_pipeline, 'metadata', {})
    
    # Step 2: Model Training
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)
    train_metadata = train_pipeline(use_prepared_data=True)
    
    # Step 3: Model Evaluation  
    print("\n" + "=" * 80)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 80)
    evaluate_pipeline()
    
    # Step 4: Generate Report
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING REPORT")
    print("=" * 80)
    report = generate_report(prep_metadata, train_metadata)
    
    # Final Summary
    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\n✅ All steps completed successfully!")
    print(f"\n📁 OUTPUT:")
    print(f"   - Trained models: {config.MODEL_SAVE_PATH}")
    print(f"   - Results & reports: {config.RESULTS_PATH}")
    print(f"   - Processed data: data/processed/")
    print(f"\n⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'preparation': prep_metadata,
        'training': train_metadata,
        'report': report
    }


if __name__ == "__main__":
    result = main_pipeline()