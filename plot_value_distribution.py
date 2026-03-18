"""
Biểu đồ phân phối Target Value (Distribution Plot)
Hiển thị phân phối giá trị cầu thủ TRƯỚC và SAU Log Transform
để chứng minh tại sao Log Transform cần thiết cho dữ liệu skewed.
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

def plot_value_distribution():
    """Vẽ biểu đồ phân phối Value trước và sau Log Transform"""
    
    # Load dữ liệu
    print("Loading data...")
    
    # Load từ cleaned data để có Value_Numeric
    data_path = 'data/cleaned_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if 'Value_Numeric' in df.columns:
            values = df['Value_Numeric'].dropna()
        else:
            print("Value_Numeric không có trong cleaned_data.csv")
            # Thử load từ raw data
            df = pd.read_csv('sofifa_players.csv')
            values = df['Value_Numeric'].dropna()
    else:
        df = pd.read_csv('sofifa_players.csv')
        values = df['Value_Numeric'].dropna()
    
    print(f"Số mẫu: {len(values)}")
    print(f"Min: €{values.min()/1e6:.2f}M, Max: €{values.max()/1e6:.2f}M")
    print(f"Mean: €{values.mean()/1e6:.2f}M, Median: €{values.median()/1e6:.2f}M")
    
    # Tính Log Transform (thêm 1 để tránh log(0))
    values_log = np.log1p(values)
    
    # Tính skewness
    skew_original = stats.skew(values)
    skew_log = stats.skew(values_log)
    
    print(f"\nSkewness trước Log Transform: {skew_original:.3f}")
    print(f"Skewness sau Log Transform: {skew_log:.3f}")
    
    # Tạo figure với 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phân phối Giá trị Cầu thủ (Value Distribution)\nTrước và Sau Log Transform', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # ====== 1. Histogram TRƯỚC Log Transform ======
    ax1 = axes[0, 0]
    ax1.hist(values / 1e6, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
    ax1.axvline(values.mean() / 1e6, color='red', linestyle='--', linewidth=2, label=f'Mean: €{values.mean()/1e6:.1f}M')
    ax1.axvline(values.median() / 1e6, color='green', linestyle='--', linewidth=2, label=f'Median: €{values.median()/1e6:.1f}M')
    ax1.set_xlabel('Giá trị (Triệu €)', fontsize=11)
    ax1.set_ylabel('Số lượng cầu thủ', fontsize=11)
    ax1.set_title(f'TRƯỚC Log Transform\nSkewness = {skew_original:.3f} (Highly Right-Skewed)', 
                  fontsize=12, fontweight='bold', color='#e74c3c')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, values.max() / 1e6 * 1.05)
    
    # Thêm annotation
    ax1.annotate('Đa số cầu thủ có\ngiá trị thấp', 
                 xy=(5, ax1.get_ylim()[1] * 0.7),
                 fontsize=10, ha='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ====== 2. Histogram SAU Log Transform ======
    ax2 = axes[0, 1]
    ax2.hist(values_log, bins=50, color='#2ecc71', edgecolor='white', alpha=0.7)
    ax2.axvline(values_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values_log.mean():.2f}')
    ax2.axvline(np.median(values_log), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values_log):.2f}')
    ax2.set_xlabel('Log(Giá trị + 1)', fontsize=11)
    ax2.set_ylabel('Số lượng cầu thủ', fontsize=11)
    ax2.set_title(f'SAU Log Transform\nSkewness = {skew_log:.3f} (Gần Normal Distribution)', 
                  fontsize=12, fontweight='bold', color='#27ae60')
    ax2.legend(loc='upper right')
    
    # Thêm annotation
    ax2.annotate('Phân phối đều hơn,\ngần Normal', 
                 xy=(ax2.get_xlim()[0] + 1, ax2.get_ylim()[1] * 0.7),
                 fontsize=10, ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ====== 3. KDE Plot so sánh ======
    ax3 = axes[1, 0]
    
    # Normalize để so sánh trên cùng scale
    values_norm = (values - values.min()) / (values.max() - values.min())
    values_log_norm = (values_log - values_log.min()) / (values_log.max() - values_log.min())
    
    sns.kdeplot(data=values_norm, ax=ax3, color='#3498db', linewidth=2, label='Original (Normalized)', fill=True, alpha=0.3)
    sns.kdeplot(data=values_log_norm, ax=ax3, color='#2ecc71', linewidth=2, label='Log Transform (Normalized)', fill=True, alpha=0.3)
    ax3.set_xlabel('Giá trị (Normalized 0-1)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('So sánh Density Plot\n(Normalized để cùng scale)', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # ====== 4. Q-Q Plot ======
    ax4 = axes[1, 1]
    
    # Q-Q plot cho log-transformed data
    stats.probplot(values_log, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Log-Transformed Data)\nĐường thẳng = Normal Distribution', fontsize=12, fontweight='bold')
    ax4.get_lines()[0].set_markerfacecolor('#2ecc71')
    ax4.get_lines()[0].set_markeredgecolor('#27ae60')
    ax4.get_lines()[1].set_color('red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'results/value_distribution_log_transform.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    # ====== Thêm 1 figure chi tiết hơn ======
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Tại sao cần Log Transform cho dữ liệu Giá trị Cầu thủ?', 
                  fontsize=14, fontweight='bold')
    
    # Box plot comparison
    ax_box = axes2[0]
    box_data = pd.DataFrame({
        'Original (€M)': values / 1e6,
        'Log Transform': values_log
    })
    
    # Vẽ 2 boxplot riêng với scale khác nhau
    ax_box.boxplot([values / 1e6], positions=[1], widths=0.6, 
                   patch_artist=True, 
                   boxprops=dict(facecolor='#3498db', alpha=0.7))
    ax_box.set_ylabel('Giá trị (Triệu €)', color='#3498db')
    ax_box.tick_params(axis='y', labelcolor='#3498db')
    ax_box.set_xlim(0.5, 2.5)
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(['Original\n(€M)', 'Log\nTransform'])
    
    ax_box2 = ax_box.twinx()
    ax_box2.boxplot([values_log], positions=[2], widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor='#2ecc71', alpha=0.7))
    ax_box2.set_ylabel('Log(Value + 1)', color='#2ecc71')
    ax_box2.tick_params(axis='y', labelcolor='#2ecc71')
    
    ax_box.set_title('Box Plot: Outliers giảm đáng kể sau Log Transform', fontsize=11, fontweight='bold')
    
    # Statistics comparison table
    ax_stats = axes2[1]
    ax_stats.axis('off')
    
    stats_data = [
        ['Metric', 'Original (€)', 'Log Transform'],
        ['Mean', f'€{values.mean()/1e6:.2f}M', f'{values_log.mean():.2f}'],
        ['Median', f'€{values.median()/1e6:.2f}M', f'{np.median(values_log):.2f}'],
        ['Std Dev', f'€{values.std()/1e6:.2f}M', f'{values_log.std():.2f}'],
        ['Skewness', f'{skew_original:.3f}', f'{skew_log:.3f}'],
        ['Kurtosis', f'{stats.kurtosis(values):.3f}', f'{stats.kurtosis(values_log):.3f}'],
        ['Min', f'€{values.min()/1e6:.2f}M', f'{values_log.min():.2f}'],
        ['Max', f'€{values.max()/1e6:.2f}M', f'{values_log.max():.2f}'],
    ]
    
    table = ax_stats.table(cellText=stats_data, loc='center', cellLoc='center',
                           colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight skewness row
    for i in range(3):
        table[(4, i)].set_facecolor('#f1c40f')
        table[(4, i)].set_text_props(fontweight='bold')
    
    ax_stats.set_title('Bảng thống kê so sánh\n(Skewness giảm từ {:.1f} xuống {:.1f})'.format(skew_original, skew_log), 
                       fontsize=11, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path2 = 'results/value_distribution_statistics.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path2}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("KẾT LUẬN: TẠI SAO CẦN LOG TRANSFORM?")
    print("="*60)
    print(f"""
1. Dữ liệu gốc có Skewness = {skew_original:.3f} (HIGHLY RIGHT-SKEWED)
   - Đa số cầu thủ có giá trị thấp (< €5M)
   - Một số ít cầu thủ có giá trị rất cao (> €100M)
   - Mean >> Median chứng minh phân phối lệch phải

2. Sau Log Transform, Skewness = {skew_log:.3f}
   - Phân phối gần Normal hơn
   - Mean ≈ Median
   - Giảm ảnh hưởng của outliers

3. Lợi ích cho ML Models:
   - Decision Trees: Splits hiệu quả hơn
   - Linear models: Giả định normality được đáp ứng
   - Gradient Boosting: Converge nhanh hơn
   - Giảm variance trong predictions
""")

if __name__ == "__main__":
    plot_value_distribution()
