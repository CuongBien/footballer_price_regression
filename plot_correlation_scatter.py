"""
Biểu đồ phân tích tương quan và mối quan hệ giữa các features
1. Correlation Heatmap - Phát hiện Multicollinearity
2. Scatter Plots - Mối quan hệ phi tuyến giữa Overall và Value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Đặt style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_correlation_heatmap():
    """Vẽ ma trận tương quan (Correlation Heatmap)"""
    
    print("="*60)
    print("1. CORRELATION HEATMAP - PHÁT HIỆN MULTICOLLINEARITY")
    print("="*60)
    
    # Load dữ liệu
    df = pd.read_csv('sofifa_players.csv')
    
    # Chọn các cột số (numeric columns) quan trọng
    # Loại bỏ các cột không cần thiết
    exclude_cols = ['Player_URL', 'Name', 'Team', 'Nationality', 'Positions', 
                    'Preferred_Foot', 'Value_Raw', 'Wage_Raw']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Tính correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # ====== Heatmap đầy đủ ======
    fig1, ax1 = plt.subplots(figsize=(18, 15))
    
    # Mask để chỉ hiển thị nửa dưới
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Vẽ heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax1)
    
    ax1.set_title('Ma trận Tương quan giữa các Features\n(Correlation Heatmap)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap_full.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/correlation_heatmap_full.png")
    
    # ====== Heatmap với Value_Numeric (Top correlations) ======
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation với Value_Numeric
    value_corr = corr_matrix['Value_Numeric'].drop('Value_Numeric').sort_values(ascending=False)
    
    # Top 15 features có correlation cao nhất với Value
    top_features = value_corr.head(15)
    bottom_features = value_corr.tail(5)
    
    # Plot top correlations
    ax2 = axes[0]
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features.values]
    bars = ax2.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features.index)
    ax2.set_xlabel('Correlation với Value_Numeric', fontsize=11)
    ax2.set_title('Top 15 Features có Tương quan cao nhất với Giá trị Cầu thủ', 
                  fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlim(-0.1, 1.0)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features.values)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot high correlation pairs (multicollinearity)
    ax3 = axes[1]
    
    # Tìm các cặp có correlation > 0.95
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    
    if len(high_corr_df) > 0:
        # Hiển thị top 15 cặp
        display_df = high_corr_df.head(15)
        
        y_pos = range(len(display_df))
        colors = ['#e74c3c' if x > 0.95 else '#f39c12' for x in display_df['Correlation']]
        bars = ax3.barh(y_pos, display_df['Correlation'], color=colors, alpha=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{row['Feature 1']}\n↔ {row['Feature 2']}" 
                            for _, row in display_df.iterrows()], fontsize=8)
        ax3.set_xlabel('Correlation', fontsize=11)
        ax3.set_title('Cặp Features có Multicollinearity (Corr > 0.90)\n[RED] > 0.95 | [ORANGE] > 0.90', 
                      fontsize=12, fontweight='bold')
        ax3.axvline(x=0.95, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.95')
        ax3.set_xlim(0.85, 1.0)
        
        # Add value labels
        for bar, val in zip(bars, display_df['Correlation']):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/correlation_with_value.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/correlation_with_value.png")
    
    # ====== Heatmap chỉ với top features ======
    fig3, ax4 = plt.subplots(figsize=(12, 10))
    
    # Chọn top features quan trọng nhất
    important_features = ['Value_Numeric', 'Overall', 'Potential', 'Wage_Numeric', 'Age',
                          'Reactions', 'Composure', 'Ball_control', 'Short_passing', 
                          'Dribbling', 'Finishing', 'Vision', 'Long_shots']
    important_features = [f for f in important_features if f in corr_matrix.columns]
    
    corr_subset = corr_matrix.loc[important_features, important_features]
    
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax4, annot_kws={"size": 9})
    
    ax4.set_title('Ma trận Tương quan - Features Quan trọng nhất\n(với Value_Numeric)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap_important.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/correlation_heatmap_important.png")
    
    # Print summary
    print(f"\n📊 Tìm thấy {len(high_corr_df)} cặp features có correlation > 0.90")
    print(f"   Trong đó {len(high_corr_df[high_corr_df['Correlation'] > 0.95])} cặp > 0.95 (cần xem xét loại bỏ)")
    
    return corr_matrix, high_corr_df


def plot_scatter_plots():
    """Vẽ Scatter Plots - Mối quan hệ phi tuyến"""
    
    print("\n" + "="*60)
    print("2. SCATTER PLOTS - MỐI QUAN HỆ PHI TUYẾN")
    print("="*60)
    
    # Load dữ liệu
    df = pd.read_csv('sofifa_players.csv')
    
    # Tạo figure với nhiều scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('Mối quan hệ giữa các Features và Giá trị Cầu thủ\n(Chứng minh tính phi tuyến - Non-linear Relationship)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # ====== 1. Overall vs Value ======
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Overall'], df['Value_Numeric']/1e6, 
                         c=df['Age'], cmap='viridis', alpha=0.5, s=20)
    ax1.set_xlabel('Overall Rating', fontsize=11)
    ax1.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax1.set_title('Overall vs Value\n(màu = Age)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Age')
    
    # Thêm trend line (polynomial)
    z = np.polyfit(df['Overall'], df['Value_Numeric']/1e6, 3)
    p = np.poly1d(z)
    x_line = np.linspace(df['Overall'].min(), df['Overall'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Polynomial fit (degree=3)')
    ax1.legend(loc='upper left')
    
    # Annotation
    ax1.annotate('Đường cong = Non-linear!\nDecision Tree xử lý tốt', 
                 xy=(55, 100), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ====== 2. Potential vs Value ======
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['Potential'], df['Value_Numeric']/1e6,
                          c=df['Overall'], cmap='plasma', alpha=0.5, s=20)
    ax2.set_xlabel('Potential Rating', fontsize=11)
    ax2.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax2.set_title('Potential vs Value\n(màu = Overall)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Overall')
    
    # Polynomial fit
    z = np.polyfit(df['Potential'], df['Value_Numeric']/1e6, 3)
    p = np.poly1d(z)
    x_line = np.linspace(df['Potential'].min(), df['Potential'].max(), 100)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # ====== 3. Age vs Value ======
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(df['Age'], df['Value_Numeric']/1e6,
                          c=df['Overall'], cmap='coolwarm', alpha=0.5, s=20)
    ax3.set_xlabel('Age', fontsize=11)
    ax3.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax3.set_title('Age vs Value\n(màu = Overall)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Overall')
    
    # Annotation cho peak age
    ax3.axvline(x=27, color='green', linestyle='--', alpha=0.7)
    ax3.annotate('Peak age ~27', xy=(27, ax3.get_ylim()[1]*0.9), fontsize=9,
                 color='green', fontweight='bold')
    
    # ====== 4. Overall vs Value (với hexbin để thấy density) ======
    ax4 = axes[1, 0]
    hb = ax4.hexbin(df['Overall'], df['Value_Numeric']/1e6, gridsize=25, 
                    cmap='YlOrRd', mincnt=1)
    ax4.set_xlabel('Overall Rating', fontsize=11)
    ax4.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax4.set_title('Overall vs Value (Density Plot)\nMàu đậm = nhiều cầu thủ', fontsize=12, fontweight='bold')
    plt.colorbar(hb, ax=ax4, label='Count')
    
    # Thêm các ngưỡng quan trọng
    for thresh in [75, 80, 85, 90]:
        ax4.axvline(x=thresh, color='blue', linestyle=':', alpha=0.5)
        
    ax4.annotate('Giá tăng EXPONENTIAL\nkhi Overall > 85', 
                 xy=(86, 50), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ====== 5. Overall vs Value với phân loại ======
    ax5 = axes[1, 1]
    
    # Chia nhóm Overall
    df['Overall_Group'] = pd.cut(df['Overall'], 
                                  bins=[0, 70, 75, 80, 85, 90, 100],
                                  labels=['<70', '70-75', '75-80', '80-85', '85-90', '90+'])
    
    # Box plot theo nhóm
    groups = ['<70', '70-75', '75-80', '80-85', '85-90', '90+']
    box_data = [df[df['Overall_Group'] == g]['Value_Numeric']/1e6 for g in groups]
    
    bp = ax5.boxplot(box_data, labels=groups, patch_artist=True)
    colors_box = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(groups)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_xlabel('Nhóm Overall Rating', fontsize=11)
    ax5.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax5.set_title('Phân phối Value theo nhóm Overall\n(Non-linear growth)', fontsize=12, fontweight='bold')
    
    # Thêm mean line
    means = [df[df['Overall_Group'] == g]['Value_Numeric'].mean()/1e6 for g in groups]
    ax5.plot(range(1, len(groups)+1), means, 'ro-', markersize=8, label='Mean')
    ax5.legend()
    
    # ====== 6. Comparison: Linear vs Non-linear fit ======
    ax6 = axes[1, 2]
    
    # Sample data để plot rõ hơn
    sample = df.sample(min(1000, len(df)), random_state=42)
    
    ax6.scatter(sample['Overall'], sample['Value_Numeric']/1e6, alpha=0.3, s=15, label='Data')
    
    # Linear fit
    z_linear = np.polyfit(df['Overall'], df['Value_Numeric']/1e6, 1)
    p_linear = np.poly1d(z_linear)
    x_line = np.linspace(45, 95, 100)
    ax6.plot(x_line, p_linear(x_line), 'b--', linewidth=2, label='Linear fit (R² thấp)')
    
    # Polynomial fit (degree 3)
    z_poly = np.polyfit(df['Overall'], df['Value_Numeric']/1e6, 3)
    p_poly = np.poly1d(z_poly)
    ax6.plot(x_line, p_poly(x_line), 'r-', linewidth=2, label='Polynomial fit (R² cao)')
    
    # Exponential-like fit
    ax6.plot(x_line, np.exp((x_line-60)/10), 'g-.', linewidth=2, label='Exponential-like')
    
    ax6.set_xlabel('Overall Rating', fontsize=11)
    ax6.set_ylabel('Giá trị (Triệu €)', fontsize=11)
    ax6.set_title('So sánh Linear vs Non-linear Fit\n→ Decision Tree/Boosting tốt hơn Linear Regression', 
                  fontsize=11, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=9)
    ax6.set_ylim(0, 150)
    
    plt.tight_layout()
    plt.savefig('results/scatter_plots_nonlinear.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/scatter_plots_nonlinear.png")
    
    # ====== Thêm 1 figure focus vào Overall vs Value ======
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df['Overall'], df['Value_Numeric']/1e6, 
                        c=df['Potential'] - df['Overall'], cmap='RdYlGn',
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    # Polynomial trend
    z = np.polyfit(df['Overall'], df['Value_Numeric']/1e6, 3)
    p = np.poly1d(z)
    x_line = np.linspace(df['Overall'].min(), df['Overall'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=3, label='Trend (Polynomial degree=3)')
    
    ax.set_xlabel('Overall Rating', fontsize=12)
    ax.set_ylabel('Giá trị Cầu thủ (Triệu €)', fontsize=12)
    ax.set_title('Mối quan hệ PHI TUYẾN giữa Overall Rating và Giá trị Cầu thủ\n' +
                 '(Màu = Potential - Overall, càng xanh = tiềm năng phát triển cao)', 
                 fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Potential - Overall\n(Room to grow)', fontsize=10)
    
    # Thêm annotations
    ax.annotate('Vùng giá trị THẤP\n(Linear growth)', 
                xy=(60, 5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.annotate('Vùng giá trị CAO\n(Exponential growth)', 
                xy=(90, 100), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.annotate('→ Decision Tree/Gradient Boosting\n   xử lý tốt mối quan hệ này\n' +
                '→ Linear Regression KHÔNG phù hợp', 
                xy=(50, 130), fontsize=11, ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(45, 100)
    ax.set_ylim(0, 180)
    
    plt.tight_layout()
    plt.savefig('results/overall_vs_value_nonlinear.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✓ Saved: results/overall_vs_value_nonlinear.png")
    
    # Print statistics
    print("\n📊 Thống kê mối quan hệ Overall vs Value:")
    
    # Correlation
    pearson_corr = df['Overall'].corr(df['Value_Numeric'])
    spearman_corr = df['Overall'].corr(df['Value_Numeric'], method='spearman')
    
    print(f"   Pearson Correlation: {pearson_corr:.4f}")
    print(f"   Spearman Correlation: {spearman_corr:.4f}")
    print(f"   (Spearman > Pearson → quan hệ phi tuyến)")
    
    # R² của linear vs polynomial
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    X = df['Overall'].values.reshape(-1, 1)
    y = df['Value_Numeric'].values
    
    # Linear
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_linear = lr.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    
    # Polynomial degree 3
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)
    y_pred_poly = lr_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    
    print(f"\n   R² Linear Regression: {r2_linear:.4f}")
    print(f"   R² Polynomial (degree=3): {r2_poly:.4f}")
    print(f"   → Polynomial tốt hơn {(r2_poly - r2_linear)/r2_linear*100:.1f}%")


def main():
    """Main function"""
    print("="*60)
    print("PHÂN TÍCH TƯƠNG QUAN VÀ MỐI QUAN HỆ PHI TUYẾN")
    print("="*60)
    
    # Chuyển vào thư mục đúng
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Correlation Heatmap
    corr_matrix, high_corr_df = plot_correlation_heatmap()
    
    # 2. Scatter Plots
    plot_scatter_plots()
    
    print("\n" + "="*60)
    print("KẾT LUẬN")
    print("="*60)
    print("""
1. MULTICOLLINEARITY:
   - Các skill GK (Diving, Handling, Kicking, Positioning, Reflexes) 
     có correlation rất cao với nhau (> 0.96)
   - Có thể gộp thành 1 feature hoặc loại bỏ bớt

2. FEATURES ẢNH HƯỞNG MẠNH NHẤT ĐẾN GIÁ TRỊ:
   - Overall Rating (~0.75)
   - Potential (~0.70)
   - Wage (~0.70)
   - Reactions, Composure, Ball Control (~0.65-0.70)

3. MỐI QUAN HỆ PHI TUYẾN:
   - Giá trị tăng EXPONENTIAL khi Overall > 85
   - Linear Regression không phù hợp (R² thấp)
   - Decision Tree/Gradient Boosting xử lý tốt

4. INSIGHTS:
   - Cầu thủ trẻ (Potential - Overall cao) có giá trị cao hơn
   - Peak age khoảng 25-28 tuổi
   - Overall > 90: Giá trị đặc biệt cao (siêu sao)
""")
    
    plt.show()


if __name__ == "__main__":
    main()
