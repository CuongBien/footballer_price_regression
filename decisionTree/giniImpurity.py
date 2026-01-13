import numpy as np
import pandas as pd

data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temp':    ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Play':    ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

def compute_Gini(target_column):
    elements, counts = np.unique(target_column, return_counts = True)
    gini = 1
    total_elements = len(target_column)
    
    for count in counts:
        p = count / total_elements
        gini -= p**2
        
    return gini

def compute_Gini_Gain(data, split_attribute_name, target_name):
    total_gini = compute_Gini(data[target_name])
    
    elements , counts = np.unique(data[split_attribute_name], return_counts = True)
    weighted_gini = 0
    total_elements = len(data[split_attribute_name])
    
    for v, count in zip(elements, counts):
        subset = data[data[split_attribute_name] == v]
        subset_gini = compute_Gini(subset[target_name])
        ratio = count / total_elements
        weighted_gini += ratio * subset_gini
        
    gg = total_gini - weighted_gini
    return gg

# --- CHẠY THỬ NGHIỆM ---

print("Dữ liệu mẫu:")
print(df)
print("-" * 30)

# Tính Entropy gốc của toàn bộ tập dữ liệu (cột Play)
parent_entropy = compute_Gini(df['Play'])
print(f"Gini ban đầu (Parent): {parent_entropy:.4f}")

# Tính IG cho thuộc tính 'Weather'
ig_weather = compute_Gini_Gain(df, 'Weather', 'Play')
print(f"==> Gini Gain của 'Weather': {ig_weather:.4f}")

# Tính IG cho thuộc tính 'Temp'
ig_temp = compute_Gini_Gain(df, 'Temp', 'Play')
print(f"==> Information Gain của 'Temp':    {ig_temp:.4f}")

print("-" * 30)
if ig_weather > ig_temp:
    print("KẾT LUẬN: Nên chọn 'Weather' làm nút gốc (Root Node) vì GG cao hơn.")
else:
    print("KẾT LUẬN: Nên chọn 'Temp' làm nút gốc (Root Node) vì GG cao hơn.")