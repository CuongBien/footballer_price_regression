"""
Script để phân tích performance và đề xuất cải thiện
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/cleaned_data.csv')
target = 'Value_Numeric'

# Phân tích target distribution
print("="*80)
print("PHÂN TÍCH TARGET VARIABLE (Value_Numeric)")
print("="*80)

print(f"\nThống kê cơ bản:")
print(f"  Mean:   €{df[target].mean():,.0f}")
print(f"  Median: €{df[target].median():,.0f}")
print(f"  Std:    €{df[target].std():,.0f}")
print(f"  Min:    €{df[target].min():,.0f}")
print(f"  Max:    €{df[target].max():,.0f}")

print(f"\nPhân vị:")
print(f"  Q1 (25%): €{df[target].quantile(0.25):,.0f}")
print(f"  Q2 (50%): €{df[target].quantile(0.50):,.0f}")
print(f"  Q3 (75%): €{df[target].quantile(0.75):,.0f}")
print(f"  Q4 (95%): €{df[target].quantile(0.95):,.0f}")

# So sánh với baseline
print("\n" + "="*80)
print("SO SÁNH VỚI BASELINE (Predict Mean)")
print("="*80)

# Load test data
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Baseline: predict mean
baseline_pred = np.full(len(y_test), df[target].mean())
baseline_mae = np.mean(np.abs(y_test - baseline_pred))
baseline_rmse = np.sqrt(np.mean((y_test - baseline_pred)**2))

print(f"\nBaseline (Predict Mean):")
print(f"  MAE:  €{baseline_mae:,.0f}")
print(f"  RMSE: €{baseline_rmse:,.0f}")

# KNN results
knn_mae = 2792763
knn_rmse = 7325713
knn_r2 = 0.7107

print(f"\nKNN (k=30, uniform):")
print(f"  MAE:  €{knn_mae:,.0f}")
print(f"  RMSE: €{knn_rmse:,.0f}")
print(f"  R2:   {knn_r2:.4f}")

print(f"\nCải thiện so với Baseline:")
print(f"  MAE:  {(1 - knn_mae/baseline_mae)*100:.1f}%")
print(f"  RMSE: {(1 - knn_rmse/baseline_rmse)*100:.1f}%")

# Tính MAPE (Mean Absolute Percentage Error)
print("\n" + "="*80)
print("MAPE (Mean Absolute Percentage Error)")
print("="*80)

# Loại bỏ giá trị 0 để tính MAPE
y_test_nonzero = y_test[y_test > 0]
if len(y_test_nonzero) > 0:
    mape = (knn_mae / y_test.mean()) * 100
    print(f"  MAPE: {mape:.1f}%")
    
    if mape < 10:
        print("  ✅ Excellent (< 10%)")
    elif mape < 20:
        print("  ✅ Good (10-20%)")
    elif mape < 30:
        print("  ⚠️ Acceptable (20-30%)")
    else:
        print("  ❌ Poor (> 30%)")

# Đề xuất cải thiện
print("\n" + "="*80)
print("ĐỀ XUẤT CẢI THIỆN")
print("="*80)

print("\n1. ✅ Log Transform Target:")
print("   - Giá trị có range rất lớn (min-max)")
print("   - Log transform sẽ giúp model học tốt hơn")
print("   - Code: y_train_log = np.log1p(y_train)")

print("\n2. 🎯 Feature Engineering:")
print("   - Tạo interaction features (Age * Overall, etc.)")
print("   - Tạo polynomial features")
print("   - Encode categorical smarter")

print("\n3. 📊 Outlier Handling:")
print("   - RMSE >> MAE → có nhiều outliers")
print("   - Consider: winsorization, remove extreme values")

print("\n4. 🔧 Model Tuning:")
print("   - KNN k=5 có MAE thấp hơn → thử k=3,7,15")
print("   - Thử metrics khác: manhattan, minkowski")
print("   - Feature selection/PCA để giảm noise")

print("\n5. 🚀 Try Other Models:")
print("   - XGBoost, LightGBM (tốt với outliers)")
print("   - Ensemble: Stacking/Blending")
print("   - Neural Networks")

print("\n" + "="*80)
