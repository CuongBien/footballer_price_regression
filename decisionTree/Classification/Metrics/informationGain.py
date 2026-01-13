import numpy as np
import pandas as pd


def compute_entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = 0
    total_elements = len(target_column)
    
    for count in counts:
        if count > 0:
            p = count / total_elements
            entropy -= p * np.log2(p)
    return entropy


def compute_information_gain(data, split_attribute_name, target_attribute_name):
    total_entropy = compute_entropy(data[target_attribute_name])
    
    elements, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = 0
    total_elements = len(data)
    
    for v, count in zip(elements, counts):
        subset = data[data[split_attribute_name] == v]
        subset_entropy = compute_entropy(subset[target_attribute_name])
        ratio = count / total_elements
        weighted_entropy += ratio * subset_entropy
        
    information_gain = total_entropy - weighted_entropy
    return information_gain

if __name__ == "__main__":
    data = {
        'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
        'Temp':    ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
        'Play':    ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    }
    
    df = pd.DataFrame(data)
    
    print("Dữ liệu mẫu:")
    print(df)
    print("-" * 30)
    
    # Tính Entropy gốc của toàn bộ tập dữ liệu (cột Play)
    parent_entropy = compute_entropy(df['Play'])
    print(f"Entropy ban đầu (Parent): {parent_entropy:.4f}")
    
    # Tính IG cho thuộc tính 'Weather'
    ig_weather = compute_information_gain(df, 'Weather', 'Play')
    print(f"==> Information Gain của 'Weather': {ig_weather:.4f}")
    
    # Tính IG cho thuộc tính 'Temp'
    ig_temp = compute_information_gain(df, 'Temp', 'Play')
    print(f"==> Information Gain của 'Temp':    {ig_temp:.4f}")
    
    print("-" * 30)
    if ig_weather > ig_temp:
        print("KẾT LUẬN: Nên chọn 'Weather' làm nút gốc (Root Node) vì IG cao hơn.")
    else:
        print("KẾT LUẬN: Nên chọn 'Temp' làm nút gốc (Root Node) vì IG cao hơn.")