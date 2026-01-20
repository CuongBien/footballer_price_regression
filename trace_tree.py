import pickle
import os
import pandas as pd
import numpy as np

os.chdir('/home/dien/Documents/DUTAI-face/team5/footballer_price_regression')

# Load MAE tree
with open('models/CustomRegressionTree_MAE.pkl', 'rb') as f:
    mae_tree = pickle.load(f)

# Load preprocessors
with open('models/preprocessors/ml_encoder.pkl', 'rb') as f:
    ml_encoder = pickle.load(f)
with open('models/preprocessors/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('models/preprocessors/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('models/preprocessors/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def preprocess(player_data):
    df = ml_encoder.transform(player_data)
    df = imputer.transform(df)
    df = encoder.transform(df)
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    extra = set(df.columns) - set(feature_names)
    if extra:
        df = df.drop(columns=list(extra))
    return df[feature_names]

# Create test data với Overall 90
overall = 90
potential = 93
player_data = pd.DataFrame({
    'Name': ['Test'],
    'Age': [25],
    'Overall': [overall],
    'Potential': [potential],
    'Height_cm': [180],
    'Weight_kg': [75],
    'Preferred_Foot': ['Right'],
    'Crossing': [overall-10],
    'Finishing': [overall],
    'Heading_accuracy': [overall-15],
    'Short_passing': [overall-5],
    'Volleys': [overall-10],
    'Dribbling': [overall],
    'Curve': [overall-10],
    'FK_Accuracy': [overall-10],
    'Long_passing': [overall-10],
    'Ball_control': [overall],
    'Acceleration': [overall-5],
    'Sprint_speed': [overall-5],
    'Agility': [overall-5],
    'Reactions': [overall],
    'Balance': [overall-5],
    'Shot_power': [overall-5],
    'Jumping': [overall-15],
    'Stamina': [overall-10],
    'Strength': [overall-15],
    'Long_shots': [overall-5],
    'Aggression': [50],
    'Interceptions': [40],
    'Standing_tackle': [35],
    'Composure': [overall],
    'Vision': [overall-5],
    'Penalties': [overall-10],
    'Positions': ['ST'],
    'GK_Diving': [10],
    'GK_Handling': [10],
    'GK_Kicking': [10],
    'GK_Positioning': [10],
    'GK_Reflexes': [10],
    'Value_Raw': ['€50M'],
    'Wage_Raw': ['€200K'],
    'Wage_Numeric': [200000],
})

df = preprocess(player_data)
X = df.values[0]

# Trace the path through tree
def trace_path(node, x, depth=0):
    indent = "  " * depth
    if node.left is None and node.right is None:
        print(f"{indent}LEAF: value={node.value/1e6:.2f}M")
        return node.value
    
    feature_val = x[node.feature]
    feature_name = feature_names[node.feature] if node.feature < len(feature_names) else f"[{node.feature}]"
    print(f"{indent}Node: {feature_name}={feature_val:.2f} <= {node.threshold:.2f}?")
    
    if feature_val <= node.threshold:
        print(f"{indent}  -> LEFT")
        return trace_path(node.left, x, depth+1)
    else:
        print(f"{indent}  -> RIGHT")
        return trace_path(node.right, x, depth+1)

print(f"Input Overall={overall}, Potential={potential}")
print(f"Tracing path through MAE tree:\n")
pred = trace_path(mae_tree.root, X)
print(f"\nPrediction: EUR{pred/1e6:.2f}M")
