import numpy as np
import pandas as pd


def compute_MAE(target_column):
    median_value = np.median(target_column)
    mae = np.mean(np.abs(target_column - median_value))
    return mae


def compute_MAE_Reduction(data, split_attribute_name, target_name):
    total_mae = compute_MAE(data[target_name])
    
    elements, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_mae = 0
    total_elements = len(data)
    
    for v, count in zip(elements, counts):
        subset = data[data[split_attribute_name] == v]
        subset_mae = compute_MAE(subset[target_name])
        ratio = count / total_elements
        weighted_mae += ratio * subset_mae
        
    mae_reduction = total_mae - weighted_mae
    return mae_reduction


# --- CHẠY THỬ NGHIỆM (chỉ khi chạy file này trực tiếp) ---
if __name__ == "__main__":
    # Tạo dữ liệu mẫu cho regression
    data = {
        'Size': ['Small', 'Small', 'Medium', 'Large', 'Large', 'Large', 'Medium', 'Small', 'Medium', 'Large'],
        'Location': ['City', 'Suburb', 'City', 'City', 'Suburb', 'Suburb', 'Suburb', 'City', 'City', 'Suburb'],
        'Price': [150, 180, 250, 400, 350, 380, 220, 170, 240, 390]  # Giá trị liên tục
    }
    
    df = pd.DataFrame(data)
    
    print("Dữ liệu mẫu (Regression):")
    print(df)
    print("-" * 50)
    
    # Tính MAE ban đầu
    parent_mae = compute_MAE(df['Price'])
    print(f"MAE ban đầu (Parent): {parent_mae:.2f}")
    
    # Tính MAE Reduction cho thuộc tính 'Size'
    mae_reduction_size = compute_MAE_Reduction(df, 'Size', 'Price')
    print(f"==> MAE Reduction của 'Size':     {mae_reduction_size:.2f}")
    
    # Tính MAE Reduction cho thuộc tính 'Location'
    mae_reduction_location = compute_MAE_Reduction(df, 'Location', 'Price')
    print(f"==> MAE Reduction của 'Location': {mae_reduction_location:.2f}")
    
    print("-" * 50)
    if mae_reduction_size > mae_reduction_location:
        print("KẾT LUẬN: Nên chọn 'Size' làm nút gốc vì MAE Reduction cao hơn.")
    else:
        print("KẾT LUẬN: Nên chọn 'Location' làm nút gốc vì MAE Reduction cao hơn.")
