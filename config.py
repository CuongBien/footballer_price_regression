"""
Configuration file cho ML Pipeline
"""

# ==================== FILE PATHS ====================
RAW_DATA_PATH = 'sofifa_players.csv'
CLEANED_DATA_PATH = 'data/cleaned_data.csv'
PROCESSED_DATA_PATH = 'data/processed_data.csv'
MODEL_SAVE_PATH = 'models/'
RESULTS_PATH = 'results/'

# ==================== DATA CLEANING ====================
MISSING_THRESHOLD = 0.5  # Xóa cột có >50% missing
LOW_VARIANCE_THRESHOLD = 0.01  # Xóa cột có variance < 0.01
HIGH_CORRELATION_THRESHOLD = 0.95  # Phát hiện multicollinearity
DUPLICATE_CHECK = True

# Các cột cần loại bỏ (không cần cho ML)
COLUMNS_TO_DROP = ['Player_URL', 'Name', 'Team', 'Nationality']

# ==================== IMPUTATION ====================
NUMERIC_IMPUTATION_STRATEGY = 'median'  # 'mean', 'median', 'mode', 'knn'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # 'most_frequent', 'constant'
KNN_NEIGHBORS = 5  # Số neighbors cho KNN Imputer

# ==================== ENCODING ====================
CATEGORICAL_ENCODING_METHOD = 'onehot'  # 'onehot', 'label', 'target'
MAX_CATEGORIES = 10  # Giới hạn số categories cho OneHot

# ==================== SCALING ====================
SCALING_METHOD = None  # Tắt scaling cho Custom Decision Tree. Options: None, 'standard', 'minmax', 'robust'

# ==================== FEATURE ENGINEERING ====================
CREATE_POLYNOMIAL_FEATURES = False
POLYNOMIAL_DEGREE = 2
FEATURE_SELECTION_METHOD = None  # Tắt feature selection: None, 'variance', 'correlation', 'model_based'
N_FEATURES_TO_SELECT = None  # None = keep all, hoặc số features muốn giữ

# ==================== TRAIN/TEST SPLIT ====================
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Từ train set
RANDOM_STATE = 42
STRATIFY_COLUMN = None  # Tên cột để stratify (cho classification)

# ==================== HANDLE IMBALANCED ====================
HANDLE_IMBALANCED = False  # True nếu là classification và data imbalanced
IMBALANCED_METHOD = 'smote'  # 'smote', 'adasyn', 'random_oversample', 'random_undersample'

# ==================== MODEL TRAINING ====================
TASK_TYPE = 'regression'  # 'regression' hoặc 'classification'
TARGET_COLUMN = 'Overall'  # Cột target

# Models để train
MODELS = {
    'regression': [
        'LinearRegression', 
        'Ridge',
        'CustomRegressionTree_MSE',   # Custom Decision Tree (MSE criterion) - TỰ LÀM
        'CustomRegressionTree_MAE',   # Custom Decision Tree (MAE criterion) - TỰ LÀM
    ],
    'classification': [
        'CustomDecisionTree_IG',      # Custom Decision Tree (Information Gain) - TỰ LÀM
        'CustomDecisionTree_Gini',    # Custom Decision Tree (Gini Impurity) - TỰ LÀM
    ]
}

# ==================== EVALUATION ====================
REGRESSION_METRICS = ['mae', 'mse', 'rmse', 'r2', 'mape']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# ==================== LOGGING ====================
VERBOSE = True
SAVE_PLOTS = True
PLOT_FORMAT = 'png'  # 'png', 'jpg', 'pdf'

# ==================== MULTI-LABEL ENCODING ====================
MULTI_LABEL_COLUMNS = ['Positions']  # Các cột có multi-label (CM, CDM, CAM, ...)
MULTI_LABEL_DELIMITER = ', '  # Delimiter giữa các labels
