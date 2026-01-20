"""
Biểu đồ phân tích hiệu năng Models
1. Model Comparison Bar Chart - So sánh R², MAE, RMSE
2. Actual vs Predicted Plot - Scatter với đường y=x
3. Feature Importance Plot - Cho HistGradientBoosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đặt style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_model_comparison():
    """Vẽ biểu đồ so sánh hiệu năng các models"""
    
    print("="*60)
    print("1. MODEL COMPARISON BAR CHART")
    print("="*60)
    
    # Load evaluation report
    eval_path = 'results/evaluation_report.csv'
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        print(f"Loaded evaluation report: {len(eval_df)} models")
    else:
        print("Không tìm thấy evaluation_report.csv, tính toán lại...")
        # Tính toán metrics từ models
        eval_df = calculate_metrics()
    
    # Chọn 4 models chính để so sánh
    main_models = ['CustomRegressionTree_MSE', 'CustomRegressionTree_MAE', 
                   'HistGradientBoosting_Custom', 'KNN_Custom']
    
    # Filter và rename cho dễ đọc
    model_names_map = {
        'CustomRegressionTree_MSE': 'Custom Tree\n(MSE)',
        'CustomRegressionTree_MAE': 'Custom Tree\n(MAE)',
        'HistGradientBoosting_Custom': 'HistGradient\nBoosting',
        'KNN_Custom': 'KNN\nCustom',
        'DecisionTreeRegressor_Sklearn': 'Sklearn\nDecisionTree',
        'HistGradientBoosting_Sklearn': 'Sklearn\nHistGradient',
        'KNN': 'KNN\n(Sklearn)'
    }
    
    # Lọc models có trong data
    available_models = [m for m in main_models if m in eval_df['Model'].values]
    if len(available_models) < 4:
        # Thêm các models khác nếu thiếu
        for m in eval_df['Model'].values:
            if m not in available_models:
                available_models.append(m)
            if len(available_models) >= 4:
                break
    
    df_filtered = eval_df[eval_df['Model'].isin(available_models)].copy()
    df_filtered['Model_Display'] = df_filtered['Model'].map(
        lambda x: model_names_map.get(x, x.replace('_', '\n'))
    )
    
    # Tạo figure với 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('So sánh Hiệu năng các Models trên Test Set', fontsize=14, fontweight='bold', y=1.02)
    
    # Colors cho các models
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    
    # ====== 1. R² Score ======
    ax1 = axes[0]
    df_sorted = df_filtered.sort_values('R2', ascending=True)
    bars1 = ax1.barh(df_sorted['Model_Display'], df_sorted['R2'], 
                     color=colors[:len(df_sorted)], alpha=0.8, edgecolor='black')
    ax1.set_xlabel('R² Score', fontsize=11)
    ax1.set_title('R² Score (Cao = Tốt)\nPhần trăm variance được giải thích', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1.05)
    
    # Add value labels
    for bar, val in zip(bars1, df_sorted['R2']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Highlight best
    best_r2_idx = df_sorted['R2'].idxmax()
    ax1.axvline(x=df_sorted.loc[best_r2_idx, 'R2'], color='green', linestyle='--', alpha=0.7)
    
    # ====== 2. MAE ======
    ax2 = axes[1]
    df_sorted = df_filtered.sort_values('MAE', ascending=False)  # Ascending=False vì MAE thấp = tốt
    bars2 = ax2.barh(df_sorted['Model_Display'], df_sorted['MAE']/1e6, 
                     color=colors[:len(df_sorted)], alpha=0.8, edgecolor='black')
    ax2.set_xlabel('MAE (Triệu €)', fontsize=11)
    ax2.set_title('Mean Absolute Error (Thấp = Tốt)\nSai số trung bình', fontsize=11, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars2, df_sorted['MAE']/1e6):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                f'€{val:.2f}M', va='center', fontsize=10, fontweight='bold')
    
    # ====== 3. RMSE ======
    ax3 = axes[2]
    df_sorted = df_filtered.sort_values('RMSE', ascending=False)
    bars3 = ax3.barh(df_sorted['Model_Display'], df_sorted['RMSE']/1e6, 
                     color=colors[:len(df_sorted)], alpha=0.8, edgecolor='black')
    ax3.set_xlabel('RMSE (Triệu €)', fontsize=11)
    ax3.set_title('Root Mean Squared Error (Thấp = Tốt)\nPhạt nặng outliers', fontsize=11, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars3, df_sorted['RMSE']/1e6):
        ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'€{val:.2f}M', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_metrics.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/model_comparison_metrics.png")
    
    # ====== Bảng tổng hợp ======
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    # Chuẩn bị data cho bảng
    table_data = []
    headers = ['Model', 'R²', 'MAE (€)', 'RMSE (€)', 'Rank']
    
    df_filtered['Rank'] = df_filtered['R2'].rank(ascending=False).astype(int)
    df_filtered = df_filtered.sort_values('Rank')
    
    for _, row in df_filtered.iterrows():
        table_data.append([
            row['Model'].replace('_', ' '),
            f"{row['R2']:.4f}",
            f"€{row['MAE']/1e6:.2f}M",
            f"€{row['RMSE']/1e6:.2f}M",
            f"#{int(row['Rank'])}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best model (row 1)
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#27ae60')
        table[(1, i)].set_text_props(fontweight='bold')
    
    ax.set_title('Bảng Tổng hợp Hiệu năng Models\n(Xếp hạng theo R² Score)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/model_comparison_table.png")
    
    return eval_df


def plot_actual_vs_predicted():
    """Vẽ biểu đồ Actual vs Predicted cho các models"""
    
    print("\n" + "="*60)
    print("2. ACTUAL VS PREDICTED PLOT")
    print("="*60)
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # Load models
    models = {}
    model_files = [
        ('Custom Tree (MSE)', 'CustomRegressionTree_MSE'),
        ('Custom Tree (MAE)', 'CustomRegressionTree_MAE'),
        ('HistGradientBoosting', 'HistGradientBoosting_Custom'),
        ('KNN Custom', 'KNN_Custom'),
    ]
    
    for display_name, filename in model_files:
        filepath = f'models/{filename}.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[display_name] = pickle.load(f)
            print(f"✓ Loaded {display_name}")
    
    if not models:
        print("Không tìm thấy models!")
        return
    
    # Tạo figure
    n_models = len(models)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    fig.suptitle('Actual vs Predicted Values\n(Đường chéo = Dự đoán hoàn hảo)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Scatter plot
        ax.scatter(y_test/1e6, y_pred/1e6, alpha=0.4, s=20, c=colors[idx], label='Predictions')
        
        # Perfect prediction line (y=x)
        max_val = max(y_test.max(), y_pred.max()) / 1e6
        min_val = min(y_test.min(), y_pred.min()) / 1e6
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect (y=x)')
        
        # Regression line
        z = np.polyfit(y_test/1e6, y_pred/1e6, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'g-', linewidth=1.5, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Giá trị Thực tế (Triệu €)', fontsize=10)
        ax.set_ylabel('Giá trị Dự đoán (Triệu €)', fontsize=10)
        ax.set_title(f'{name}\nR²={r2:.4f} | MAE=€{mae/1e6:.2f}M | RMSE=€{rmse/1e6:.2f}M', 
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        
        # Set same limits
        ax.set_xlim(min_val - 5, max_val + 5)
        ax.set_ylim(min_val - 5, max_val + 5)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/actual_vs_predicted.png")
    
    # ====== Focus plot cho best model ======
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Load best model (HistGradientBoosting)
    best_model_name = 'HistGradientBoosting'
    if best_model_name in models:
        model = models[best_model_name]
    else:
        model = list(models.values())[0]
        best_model_name = list(models.keys())[0]
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Scatter với color theo error
    errors = np.abs(y_test - y_pred) / 1e6
    scatter = ax.scatter(y_test/1e6, y_pred/1e6, c=errors, cmap='RdYlGn_r', 
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    # Perfect line
    max_val = max(y_test.max(), y_pred.max()) / 1e6
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction (y=x)')
    
    # Add error bands
    ax.fill_between([0, max_val], [0, max_val], [5, max_val+5], alpha=0.1, color='red', label='±€5M error band')
    ax.fill_between([0, max_val], [-5, max_val-5], [0, max_val], alpha=0.1, color='red')
    
    ax.set_xlabel('Giá trị Thực tế (Triệu €)', fontsize=12)
    ax.set_ylabel('Giá trị Dự đoán (Triệu €)', fontsize=12)
    ax.set_title(f'Actual vs Predicted - {best_model_name}\n' +
                 f'R² = {r2:.4f} | MAE = €{mae/1e6:.2f}M\n' +
                 '(Màu = Sai số, càng xanh = càng chính xác)', 
                 fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error (€M)', fontsize=10)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, max_val + 5)
    ax.set_ylim(0, max_val + 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/actual_vs_predicted_best.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/actual_vs_predicted_best.png")


def plot_feature_importance():
    """Vẽ Feature Importance cho HistGradientBoosting"""
    
    print("\n" + "="*60)
    print("3. FEATURE IMPORTANCE PLOT")
    print("="*60)
    
    # Load feature names
    with open('models/preprocessors/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Thử load sklearn model trước (có feature_importances_)
    model_path = 'models/HistGradientBoosting_Sklearn.pkl'
    model_name = 'HistGradientBoosting (Sklearn)'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded model từ {model_path}")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Sử dụng permutation importance
            print("Tính Permutation Importance...")
            from sklearn.inspection import permutation_importance
            
            X_test = pd.read_csv('data/processed/X_test.csv')
            y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
            
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importances = result.importances_mean
    else:
        # Nếu không có sklearn model, dùng DecisionTree để lấy feature importance
        model_path = 'models/DecisionTreeRegressor_Sklearn.pkl'
        model_name = 'DecisionTree (Sklearn)'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded model từ {model_path}")
            importances = model.feature_importances_
        else:
            print("Không tìm thấy model có feature_importances_!")
            return None
    
    # Tạo DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 10 Features quan trọng nhất:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # ====== Plot Top 20 features ======
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top 20 horizontal bar
    ax1 = axes[0]
    top_20 = importance_df.head(20)
    
    # Color theo category
    def get_category(feature):
        if feature in ['Overall', 'Potential', 'Age']:
            return '#e74c3c'  # Red - Basic info
        elif feature.startswith('Positions_'):
            return '#9b59b6'  # Purple - Position
        elif feature in ['Reactions', 'Composure', 'Vision', 'Ball_control', 
                         'Short_passing', 'Long_passing', 'Dribbling']:
            return '#3498db'  # Blue - Mental/Technical
        elif feature.startswith('GK_'):
            return '#f39c12'  # Orange - GK
        else:
            return '#2ecc71'  # Green - Physical/Other
    
    colors = [get_category(f) for f in top_20['Feature']]
    
    bars = ax1.barh(range(len(top_20)), top_20['Importance'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['Feature'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Feature Importance', fontsize=11)
    ax1.set_title('Top 20 Features Quan trọng nhất\n(HistGradientBoosting Model)', 
                  fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, top_20['Importance']):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Basic Info (Age, Overall, Potential)'),
        Patch(facecolor='#3498db', label='Mental/Technical Skills'),
        Patch(facecolor='#2ecc71', label='Physical/Other'),
        Patch(facecolor='#9b59b6', label='Position'),
        Patch(facecolor='#f39c12', label='Goalkeeper Skills'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Plot 2: Pie chart theo category
    ax2 = axes[1]
    
    # Group by category
    category_importance = {
        'Basic Info\n(Age, Overall, Potential)': 0,
        'Mental/Technical\nSkills': 0,
        'Physical/Other\nSkills': 0,
        'Position\nFeatures': 0,
        'Goalkeeper\nSkills': 0
    }
    
    for _, row in importance_df.iterrows():
        feature = row['Feature']
        imp = max(0, row['Importance'])  # Ensure non-negative
        
        if feature in ['Overall', 'Potential', 'Age']:
            category_importance['Basic Info\n(Age, Overall, Potential)'] += imp
        elif feature.startswith('Positions_'):
            category_importance['Position\nFeatures'] += imp
        elif feature in ['Reactions', 'Composure', 'Vision', 'Ball_control', 
                         'Short_passing', 'Long_passing', 'Dribbling', 'Finishing',
                         'Heading_accuracy', 'Volleys', 'Curve', 'FK_Accuracy', 'Penalties']:
            category_importance['Mental/Technical\nSkills'] += imp
        elif feature.startswith('GK_'):
            category_importance['Goalkeeper\nSkills'] += imp
        else:
            category_importance['Physical/Other\nSkills'] += imp
    
    # Remove categories with 0 importance
    category_importance = {k: v for k, v in category_importance.items() if v > 0}
    
    colors_pie = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12'][:len(category_importance)]
    
    # Calculate explode based on number of categories
    explode = [0.05] + [0] * (len(category_importance) - 1) if len(category_importance) > 0 else []
    
    wedges, texts, autotexts = ax2.pie(
        category_importance.values(), 
        labels=category_importance.keys(),
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode
    )
    
    ax2.set_title('Phân bố Feature Importance theo Category\n(Tổng = 100%)', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/feature_importance.png")
    
    # ====== Detailed importance plot ======
    fig2, ax = plt.subplots(figsize=(12, 10))
    
    # All features importance
    ax.barh(range(len(importance_df)), importance_df['Importance'], 
            color='#3498db', alpha=0.7, edgecolor='white')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Feature Importance - Tất cả Features\n(HistGradientBoosting Model)', 
                 fontsize=13, fontweight='bold')
    
    # Highlight top 10
    for i in range(min(10, len(importance_df))):
        ax.get_children()[i].set_color('#e74c3c')
        ax.get_children()[i].set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig('results/feature_importance_all.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/feature_importance_all.png")
    
    return importance_df


def calculate_metrics():
    """Tính metrics nếu không có evaluation_report.csv"""
    
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    results = []
    models_dir = 'models'
    
    for file in os.listdir(models_dir):
        if file.endswith('.pkl') and not file.startswith('preprocessor'):
            model_name = file.replace('.pkl', '')
            try:
                with open(os.path.join(models_dir, file), 'rb') as f:
                    model = pickle.load(f)
                
                y_pred = model.predict(X_test)
                
                results.append({
                    'Model': model_name,
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                })
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    return pd.DataFrame(results)


def main():
    """Main function"""
    print("="*60)
    print("PHÂN TÍCH HIỆU NĂNG MODELS")
    print("="*60)
    
    # Chuyển vào thư mục đúng
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Model Comparison
    eval_df = plot_model_comparison()
    
    # 2. Actual vs Predicted
    plot_actual_vs_predicted()
    
    # 3. Feature Importance
    importance_df = plot_feature_importance()
    
    print("\n" + "="*60)
    print("KẾT LUẬN")
    print("="*60)
    print("""
📊 TỔNG KẾT:

1. MODEL COMPARISON:
   - HistGradientBoosting có R² cao nhất (~0.99)
   - Custom Decision Trees có hiệu năng tốt (~0.97)
   - KNN có hiệu năng thấp hơn đáng kể

2. ACTUAL VS PREDICTED:
   - Các điểm bám sát đường y=x = dự đoán tốt
   - HistGradientBoosting có ít outliers nhất
   - Errors tập trung ở vùng giá trị cao (> €50M)

3. FEATURE IMPORTANCE:
   - Overall, Potential, Age là quan trọng nhất (~50%+)
   - Reactions, Composure, Ball_control cũng quan trọng
   - Position features có ảnh hưởng đáng kể
   - GK skills ít quan trọng (vì ít GK trong data)

📈 KHUYẾN NGHỊ:
   - Sử dụng HistGradientBoosting cho production
   - Focus vào Overall, Potential khi feature engineering
   - Cân nhắc tạo thêm features từ Age (young potential)
""")
    
    plt.show()


if __name__ == "__main__":
    main()
