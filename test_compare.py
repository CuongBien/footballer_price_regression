import pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/dien/Documents/DUTAI-face/team5/footballer_price_regression')

# Load models trực tiếp
with open('models/CustomRegressionTree_MSE.pkl', 'rb') as f:
    custom_mse = pickle.load(f)
with open('models/CustomRegressionTree_MAE.pkl', 'rb') as f:
    custom_mae = pickle.load(f)
with open('models/DecisionTreeRegressor_Sklearn.pkl', 'rb') as f:
    sklearn_tree = pickle.load(f)

# Load preprocessors
with open('models/preprocessors/ml_encoder.pkl', 'rb') as f:
    ml_encoder = pickle.load(f)
with open('models/preprocessors/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('models/preprocessors/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('models/preprocessors/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def preprocess_and_predict(overall, potential):
    positions_str = "ST"
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
        'Positions': [positions_str],
        'GK_Diving': [10],
        'GK_Handling': [10],
        'GK_Kicking': [10],
        'GK_Positioning': [10],
        'GK_Reflexes': [10],
        'Value_Raw': ['€50M'],
        'Wage_Raw': ['€200K'],
        'Wage_Numeric': [200000],
    })
    
    # Preprocess
    df = ml_encoder.transform(player_data)
    df = imputer.transform(df)
    df = encoder.transform(df)
    
    # Align features
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    extra = set(df.columns) - set(feature_names)
    if extra:
        df = df.drop(columns=list(extra))
    df = df[feature_names]
    
    return {
        'Custom MSE': custom_mse.predict(df)[0]/1e6,
        'Custom MAE': custom_mae.predict(df)[0]/1e6,
        'Sklearn': sklearn_tree.predict(df)[0]/1e6
    }

print("=" * 70)
print("TEST VỚI CÁC MỨC OVERALL/POTENTIAL KHÁC NHAU")
print("=" * 70)
print(f"{'Overall':>8} | {'Potential':>9} | {'Custom MSE':>12} | {'Custom MAE':>12} | {'Sklearn':>12} | {'Chênh lệch':>12}")
print("-" * 70)

for overall in [70, 75, 80, 85, 88, 90, 92, 94]:
    potential = overall + 3
    r = preprocess_and_predict(overall, potential)
    diff = abs(r['Custom MSE'] - r['Sklearn'])
    pct = diff/max(r['Sklearn'], 0.01)*100
    print(f"{overall:>8} | {potential:>9} | €{r['Custom MSE']:>9.2f}M | €{r['Custom MAE']:>9.2f}M | €{r['Sklearn']:>9.2f}M | €{diff:>7.2f}M ({pct:>4.0f}%)")
