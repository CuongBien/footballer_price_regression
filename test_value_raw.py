import pickle
import os
import pandas as pd

os.chdir('/home/dien/Documents/DUTAI-face/team5/footballer_price_regression')

# Load models và preprocessors
with open('models/CustomRegressionTree_MSE.pkl', 'rb') as f:
    custom_mse = pickle.load(f)
with open('models/CustomRegressionTree_MAE.pkl', 'rb') as f:
    custom_mae = pickle.load(f)
with open('models/DecisionTreeRegressor_Sklearn.pkl', 'rb') as f:
    sklearn_tree = pickle.load(f)

with open('models/preprocessors/ml_encoder.pkl', 'rb') as f:
    ml_encoder = pickle.load(f)
with open('models/preprocessors/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('models/preprocessors/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('models/preprocessors/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def preprocess_and_predict(overall, potential, value_raw):
    positions_str = 'ST'
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
        'Value_Raw': [value_raw],  
        'Wage_Raw': ['€200K'],
        'Wage_Numeric': [200000],
    })
    
    df = ml_encoder.transform(player_data)
    df = imputer.transform(df)
    df = encoder.transform(df)
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

print('Test Overall=90, Potential=93 với các Value_Raw khác nhau:')
print('-' * 70)
for value_raw in ['€5M', '€10M', '€30M', '€50M', '€80M', '€100M', '€130M', '€150M']:
    r = preprocess_and_predict(90, 93, value_raw)
    print(f'Value_Raw={value_raw:>8}: MSE=€{r["Custom MSE"]:>6.2f}M, MAE=€{r["Custom MAE"]:>6.2f}M, Sklearn=€{r["Sklearn"]:>6.2f}M')
