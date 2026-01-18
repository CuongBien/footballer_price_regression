"""
MAIN ML PIPELINE - FOOTBALLER RATING PREDICTION
Chạy toàn bộ pipeline: Data Preparation → Training → Evaluation
"""

import warnings
warnings.filterwarnings('ignore')

from data_preparation_pipeline import DataPreparationPipeline
from train_pipeline import train_pipeline
from evaluate_pipeline import evaluate_pipeline


def main_pipeline():
    """Main pipeline thực hiện toàn bộ quy trình ML"""
    
    # Data Preparation
    prep_pipeline = DataPreparationPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = prep_pipeline.run_full_pipeline()
    
    # Model Training
    metadata = train_pipeline(use_prepared_data=True)
    
    # Model Evaluation
    evaluate_pipeline()
    
    return {
        'preparation': prep_pipeline.metadata,
        'training': metadata
    }


if __name__ == "__main__":
    result = main_pipeline()