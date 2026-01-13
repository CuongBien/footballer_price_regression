import numpy as np
import pandas as pd


def compute_MSE(target_column):
    mean_value = np.mean(target_column)
    mse = np.mean((target_column - mean_value) ** 2)
    return mse


def compute_MSE_Reduction(data, split_attribute_name, target_name):
    total_mse = compute_MSE(data[target_name])
    
    elements, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_mse = 0
    total_elements = len(data)
    
    for v, count in zip(elements, counts):
        subset = data[data[split_attribute_name] == v]
        subset_mse = compute_MSE(subset[target_name])
        ratio = count / total_elements
        weighted_mse += ratio * subset_mse
        
    mse_reduction = total_mse - weighted_mse
    return mse_reduction


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
    
    # Tính MSE ban đầu
    parent_mse = compute_MSE(df['Price'])
    print(f"MSE ban đầu (Parent): {parent_mse:.2f}")
    
    # Tính MSE Reduction cho thuộc tính 'Size'
    mse_reduction_size = compute_MSE_Reduction(df, 'Size', 'Price')
    print(f"==> MSE Reduction của 'Size':     {mse_reduction_size:.2f}")
    
    # Tính MSE Reduction cho thuộc tính 'Location'
    mse_reduction_location = compute_MSE_Reduction(df, 'Location', 'Price')
    print(f"==> MSE Reduction của 'Location': {mse_reduction_location:.2f}")
    
    print("-" * 50)
    if mse_reduction_size > mse_reduction_location:
        print("KẾT LUẬN: Nên chọn 'Size' làm nút gốc vì MSE Reduction cao hơn.")
    else:
        print("KẾT LUẬN: Nên chọn 'Location' làm nút gốc vì MSE Reduction cao hơn.")
