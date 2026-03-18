"""
STREAMLIT APP - FOOTBALLER VALUE PREDICTION
Giao diện web để dự đoán giá trị cầu thủ
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# Đảm bảo working directory đúng
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from inference_pipeline import ModelInference
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Footballer Value Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #ff7f0e;
        text-align: center;
        padding: 2rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_models():
    """Load tất cả models và metadata"""
    models = {}
    model_names = [
        'CustomRegressionTree_MSE', 
        'CustomRegressionTree_MAE', 
        'DecisionTreeRegressor_Sklearn',
        'HistGradientBoosting_Custom', 
        'HistGradientBoosting_Sklearn',
        'KNN_Custom'
    ]
    
    for model_name in model_names:
        model_path = f'models/{model_name}.pkl'
        if os.path.exists(model_path):
            try:
                print("=" * 70)
                print("LOADING MODEL:", model_name)
                print("=" * 70)
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Không thể load model {model_name}: {e}")
    
    return models

@st.cache_data
def load_sample_data():
    """Load sample data"""
    if os.path.exists('sofifa_players.csv'):
        df = pd.read_csv('sofifa_players.csv', nrows=100)
        return df
    return None

def format_currency(value):
    """Format số thành currency"""
    if value >= 1_000_000:
        return f"€{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"€{value/1_000:.1f}K"
    else:
        return f"€{value:.0f}"

def create_comparison_chart(predictions):
    """Tạo biểu đồ so sánh predictions"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[format_currency(v) for v in predictions.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Dự đoán giá trị từ các models",
        xaxis_title="Model",
        yaxis_title="Giá trị (€)",
        height=400,
        showlegend=False
    )
    
    return fig

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<div class="main-header">⚽ Footballer Value Prediction</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/soccer-ball.png", width=80)
        st.title("⚙️ Cấu hình")
        
        mode = st.radio(
            "Chọn chế độ:",
            ["📊 Batch Prediction", "✏️ Manual Input", "📈 Model Comparison"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Thông tin Models")
        
        # Load models info
        if os.path.exists('models/training_scores.pkl'):
            with open('models/training_scores.pkl', 'rb') as f:
                scores = pickle.load(f)
                st.info(f"Số models đã train: {len(scores)}")
        
        st.markdown("---")
        st.markdown("### 📖 Hướng dẫn")
        st.markdown("""
        1. **Batch Prediction**: Upload file CSV để dự đoán hàng loạt
        2. **Manual Input**: Nhập thông tin cầu thủ thủ công
        3. **Model Comparison**: So sánh hiệu suất các models
        """)
    
    # Main content based on mode
    if mode == "📊 Batch Prediction":
        batch_prediction_page()
    elif mode == "✏️ Manual Input":
        manual_input_page()
    else:
        model_comparison_page()

# ==================== BATCH PREDICTION PAGE ====================

def batch_prediction_page():
    st.markdown('<div class="sub-header">📊 Batch Prediction - Dự đoán hàng loạt</div>', 
                unsafe_allow_html=True)
    
    # Lấy danh sách models có sẵn
    available_models = []
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and not file.startswith('preprocessor') and not file.startswith('log_'):
                model_name = file.replace('.pkl', '')
                available_models.append(model_name)
    
    # Nếu không tìm thấy, dùng default
    if not available_models:
        available_models = [
            'CustomRegressionTree_MSE', 
            'CustomRegressionTree_MAE',
            'DecisionTreeRegressor_Sklearn',
            'CustomRandomForestRegressor', 
            'HistGradientBoosting_Custom',
            'HistGradientBoosting_Sklearn',
            'KNN'
        ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file CSV chứa thông tin cầu thủ",
            type=['csv'],
            help="File CSV phải có các cột tương tự dữ liệu training"
        )
    
    with col2:
        model_choice = st.selectbox(
            "Chọn model:",
            available_models
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Đã load {len(df)} cầu thủ")
            
            # Hiển thị preview
            with st.expander("👀 Xem dữ liệu (5 dòng đầu)"):
                st.dataframe(df.head())
            
            if st.button("🚀 Bắt đầu dự đoán", type="primary"):
                with st.spinner("Đang dự đoán..."):
                    try:
                        # Load model và predict
                        inferencer = ModelInference(model_name=model_choice)
                        predictions = inferencer.predict(df)
                        
                        # Thêm predictions vào dataframe
                        df['Predicted_Value'] = predictions
                        df['Predicted_Value_Formatted'] = df['Predicted_Value'].apply(format_currency)
                        
                        st.success("✓ Dự đoán hoàn tất!")
                        
                        # Hiển thị kết quả
                        st.markdown("### 📊 Kết quả dự đoán")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Trung bình", format_currency(predictions.mean()))
                        with col2:
                            st.metric("Cao nhất", format_currency(predictions.max()))
                        with col3:
                            st.metric("Thấp nhất", format_currency(predictions.min()))
                        with col4:
                            st.metric("Độ lệch chuẩn", format_currency(predictions.std()))
                        
                        # Hiển thị bảng
                        if 'Name' in df.columns:
                            display_cols = ['Name', 'Age', 'Overall', 'Potential', 'Predicted_Value_Formatted']
                            display_df = df[[col for col in display_cols if col in df.columns]]
                        else:
                            display_df = df
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download kết quả (CSV)",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Biểu đồ phân phối
                        fig = px.histogram(
                            df, x='Predicted_Value',
                            title="Phân phối giá trị dự đoán",
                            nbins=30,
                            labels={'Predicted_Value': 'Giá trị (€)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
        
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
    
    else:
        # Hiển thị sample data
        st.info("💡 Bạn có thể download sample data để test:")
        sample_df = load_sample_data()
        if sample_df is not None:
            csv = sample_df.head(10).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name="sample_data.csv",
                mime="text/csv"
            )

# ==================== MANUAL INPUT PAGE ====================

def manual_input_page():
    st.markdown('<div class="sub-header">✏️ Manual Input - Nhập thông tin cầu thủ</div>', 
                unsafe_allow_html=True)
    
    st.info("📝 Nhập thông tin cầu thủ để dự đoán giá trị")
    
    # Lấy danh sách models có sẵn VÀ cho phép chọn ngay từ đầu
    available_models = []
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and not file.startswith('preprocessor') and not file.startswith('log_'):
                model_name = file.replace('.pkl', '')
                available_models.append(model_name)
    
    # Nếu không tìm thấy models, dùng default list
    if not available_models:
        available_models = [
            'CustomRegressionTree_MSE', 
            'CustomRegressionTree_MAE',
            'DecisionTreeRegressor_Sklearn',
            'HistGradientBoosting_Custom', 
            'HistGradientBoosting_Sklearn'
        ]
    
    # Chọn models TRƯỚC KHI nhập data
    st.markdown("### 🎯 Chọn Models để Dự Đoán")
    selected_models = st.multiselect(
        "Chọn 1 hoặc nhiều models (càng nhiều càng chính xác với ensemble):",
        options=available_models,
        default=available_models[:min(3, len(available_models))],
        help="Models sẽ dự đoán và kết hợp kết quả để cho ra giá trị tối ưu"
    )
    
    if not selected_models:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 model để tiếp tục!")
        return
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Thông tin cơ bản")
        name = st.text_input("Tên cầu thủ", "Lionel Messi")
        age = st.slider("Tuổi", 16, 45, 25)
        overall = st.slider("Overall Rating", 40, 99, 85)
        potential = st.slider("Potential", 40, 99, 90)
        height = st.number_input("Chiều cao (cm)", 150, 210, 175)
        weight = st.number_input("Cân nặng (kg)", 50, 120, 75)
        preferred_foot = st.selectbox("Chân thuận", ["Right", "Left"])
        positions = st.multiselect("Vị trí", ["ST", "CF", "LW", "RW", "CAM", "CM", "CDM", "CB", "LB", "RB", "GK"], 
                                   default=["ST"])
    
    with col2:
        st.markdown("#### ⚽ Kỹ năng tấn công")
        crossing = st.slider("Crossing", 0, 99, 75)
        finishing = st.slider("Finishing", 0, 99, 80)
        heading = st.slider("Heading Accuracy", 0, 99, 70)
        short_passing = st.slider("Short Passing", 0, 99, 85)
        volleys = st.slider("Volleys", 0, 99, 75)
        dribbling = st.slider("Dribbling", 0, 99, 90)
        curve = st.slider("Curve", 0, 99, 80)
        fk_accuracy = st.slider("Free Kick Accuracy", 0, 99, 80)
        long_passing = st.slider("Long Passing", 0, 99, 80)
        ball_control = st.slider("Ball Control", 0, 99, 90)
    
    with col3:
        st.markdown("#### 🏃 Thể chất & Phòng thủ")
        acceleration = st.slider("Acceleration", 0, 99, 85)
        sprint_speed = st.slider("Sprint Speed", 0, 99, 85)
        agility = st.slider("Agility", 0, 99, 85)
        reactions = st.slider("Reactions", 0, 99, 90)
        balance = st.slider("Balance", 0, 99, 85)
        shot_power = st.slider("Shot Power", 0, 99, 85)
        jumping = st.slider("Jumping", 0, 99, 75)
        stamina = st.slider("Stamina", 0, 99, 80)
        strength = st.slider("Strength", 0, 99, 70)
        long_shots = st.slider("Long Shots", 0, 99, 85)
        aggression = st.slider("Aggression", 0, 99, 45)
        interceptions = st.slider("Interceptions", 0, 99, 40)
        standing_tackle = st.slider("Standing Tackle", 0, 99, 35)
        composure = st.slider("Composure", 0, 99, 95)
        vision = st.slider("Vision", 0, 99, 90)
        penalties = st.slider("Penalties", 0, 99, 80)
    
    st.markdown("---")
    
    if st.button("🎯 Dự đoán giá trị", type="primary", use_container_width=True):
        with st.spinner("Đang tính toán..."):
            try:
                # Tạo DataFrame với tất cả features cần thiết (giống raw data)
                positions_str = ", ".join(positions)
                
                player_data = pd.DataFrame({
                    'Name': [name],
                    'Age': [age],
                    'Overall': [overall],
                    'Potential': [potential],
                    'Height_cm': [height],
                    'Weight_kg': [weight],
                    'Preferred_Foot': [preferred_foot],
                    'Crossing': [crossing],
                    'Finishing': [finishing],
                    'Heading_accuracy': [heading],
                    'Short_passing': [short_passing],
                    'Volleys': [volleys],
                    'Dribbling': [dribbling],
                    'Curve': [curve],
                    'FK_Accuracy': [fk_accuracy],
                    'Long_passing': [long_passing],
                    'Ball_control': [ball_control],
                    'Acceleration': [acceleration],
                    'Sprint_speed': [sprint_speed],
                    'Agility': [agility],
                    'Reactions': [reactions],
                    'Balance': [balance],
                    'Shot_power': [shot_power],
                    'Jumping': [jumping],
                    'Stamina': [stamina],
                    'Strength': [strength],
                    'Long_shots': [long_shots],
                    'Aggression': [aggression],
                    'Interceptions': [interceptions],
                    'Standing_tackle': [standing_tackle],
                    'Composure': [composure],
                    'Vision': [vision],
                    'Penalties': [penalties],
                    'Positions': [positions_str],
                    # Thêm các GK skills (default cho non-GK)
                    'GK_Diving': [50 if 'GK' in positions else 10],
                    'GK_Handling': [50 if 'GK' in positions else 10],
                    'GK_Kicking': [50 if 'GK' in positions else 10],
                    'GK_Positioning': [50 if 'GK' in positions else 10],
                    'GK_Reflexes': [50 if 'GK' in positions else 10],
                    # NOTE: Value_Raw, Wage_Raw, Wage_Numeric đã bị loại bỏ 
                    # khỏi features để tránh data leakage
                })
                
                # Predict với các models đã chọn (selected_models đã được define ở đầu page)
                predictions = {}
                
                for model_name in selected_models:
                    try:
                        inferencer = ModelInference(model_name=model_name)
                        pred = inferencer.predict(player_data)[0]
                        predictions[model_name] = pred
                    except Exception as e:
                        st.warning(f"Không thể dự đoán với {model_name}: {str(e)[:100]}")
                
                if predictions:
                    # Load model scores để tính weighted average
                    model_weights = {}
                    if os.path.exists('results/evaluation_report.csv'):
                        eval_df = pd.read_csv('results/evaluation_report.csv')
                        for model_name in predictions.keys():
                            model_row = eval_df[eval_df['Model'] == model_name]
                            if not model_row.empty and 'R2' in eval_df.columns:
                                r2_score = model_row['R2'].values[0]
                                # Chỉ dùng models có R2 > 0
                                if r2_score > 0:
                                    model_weights[model_name] = r2_score
                    
                    # Nếu không có weights, dùng equal weights
                    if not model_weights:
                        model_weights = {k: 1.0 for k in predictions.keys()}
                    
                    # Tính weighted average (ưu tiên models tốt hơn)
                    weighted_sum = sum(predictions[k] * model_weights.get(k, 0) for k in predictions.keys())
                    weight_total = sum(model_weights.get(k, 0) for k in predictions.keys())
                    weighted_avg = weighted_sum / weight_total if weight_total > 0 else np.mean(list(predictions.values()))
                    
                    # Tính median (robust với outliers)
                    median_prediction = np.median(list(predictions.values()))
                    
                    # Lọc outliers (loại bỏ predictions quá xa median)
                    pred_values = np.array(list(predictions.values()))
                    q1, q3 = np.percentile(pred_values, [25, 75])
                    iqr = q3 - q1
                    filtered_preds = pred_values[(pred_values >= q1 - 1.5*iqr) & (pred_values <= q3 + 1.5*iqr)]
                    robust_avg = np.mean(filtered_preds) if len(filtered_preds) > 0 else median_prediction
                    
                    # Hiển thị kết quả với 3 metrics
                    st.markdown("### 💰 Giá trị dự đoán")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Weighted Average (Khuyến nghị)", 
                            format_currency(weighted_avg),
                            help="Trung bình có trọng số dựa trên hiệu suất models"
                        )
                    
                    with col2:
                        st.metric(
                            "Median (An toàn)", 
                            format_currency(median_prediction),
                            help="Giá trị trung vị, ít bị ảnh hưởng bởi outliers"
                        )
                    
                    with col3:
                        st.metric(
                            "Robust Average", 
                            format_currency(robust_avg),
                            help="Trung bình sau khi loại bỏ outliers"
                        )
                    
                    # Hiển thị độ phân tán
                    std_dev = np.std(list(predictions.values()))
                    cv = (std_dev / np.mean(list(predictions.values()))) * 100  # Coefficient of variation
                    
                    if cv > 50:
                        st.warning(f"⚠️ Độ phân tán cao ({cv:.1f}%) - Models dự đoán rất khác nhau. Nên retrain hoặc kiểm tra data.")
                    elif cv > 30:
                        st.info(f"ℹ️ Độ phân tán vừa phải ({cv:.1f}%) - Kết quả có thể chấp nhận được.")
                    else:
                        st.success(f"✓ Độ phân tán thấp ({cv:.1f}%) - Models dự đoán khá nhất quán.")
                    
                    # Biểu đồ so sánh
                    st.plotly_chart(create_comparison_chart(predictions), use_container_width=True)
                    
                    # Bảng chi tiết với weights
                    st.markdown("### 📊 Chi tiết dự đoán từng model")
                    results_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Predicted Value': [format_currency(v) for v in predictions.values()],
                        'Raw Value': list(predictions.values()),
                        'Model Weight': [f"{model_weights.get(k, 0):.3f}" for k in predictions.keys()],
                        'Độ lệch từ Median': [f"{((v - median_prediction) / median_prediction * 100):.1f}%" 
                                              for v in predictions.values()]
                    })
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("### 💡 Khuyến nghị")
                    st.info(f"""
                    **Giá trị ước tính tốt nhất:** {format_currency(weighted_avg)}
                    
                    - Dựa trên weighted average của các models với trọng số tương ứng với R² score
                    - Nếu không chắc chắn, có thể tham khảo giá trị Median: {format_currency(median_prediction)}
                    - Khoảng dao động: {format_currency(min(predictions.values()))} - {format_currency(max(predictions.values()))}
                    """)
                else:
                    st.error("Không có model nào hoạt động")
                    
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
                st.exception(e)

# ==================== MODEL COMPARISON PAGE ====================

def model_comparison_page():
    st.markdown('<div class="sub-header">📈 Model Comparison - So sánh các models</div>', 
                unsafe_allow_html=True)
    
    # Load scores
    if os.path.exists('models/training_scores.pkl'):
        with open('models/training_scores.pkl', 'rb') as f:
            scores = pickle.load(f)
        
        if os.path.exists('results/evaluation_report.csv'):
            eval_df = pd.read_csv('results/evaluation_report.csv')
            
            st.markdown("### 📊 Training Scores")
            # Handle scores - could be dict of numbers or dict of dicts
            scores_list = []
            for model_name, score_value in scores.items():
                if isinstance(score_value, dict):
                    # If it's a dict, try to get train score
                    train_score = score_value.get('train', score_value.get('r2', 0))
                else:
                    # If it's a number, use directly
                    train_score = score_value
                scores_list.append(f"{train_score*100:.2f}%")
            
            scores_df = pd.DataFrame({
                'Model': list(scores.keys()),
                'Train R² Score': scores_list
            })
            st.dataframe(scores_df, use_container_width=True)
            
            st.markdown("### 📈 Test Performance")
            
            # Clean infinity and nan values trước khi hiển thị
            eval_df_display = eval_df.copy()
            for col in ['MAE', 'RMSE', 'MAPE']:
                if col in eval_df_display.columns:
                    # Replace inf with "N/A"
                    eval_df_display[col] = eval_df_display[col].replace([np.inf, -np.inf], np.nan)
                    # Format numbers or show N/A
                    eval_df_display[col] = eval_df_display[col].apply(
                        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                    )
            
            if 'R2' in eval_df_display.columns:
                eval_df_display['R2'] = eval_df_display['R2'].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and not np.isinf(x) else "N/A"
                )
            
            st.dataframe(eval_df_display, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if 'R2' in eval_df.columns:
                    fig = px.bar(
                        eval_df, x='Model', y='R2',
                        title="R² Score Comparison",
                        color='R2',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'MAE' in eval_df.columns:
                    fig = px.bar(
                        eval_df, x='Model', y='MAE',
                        title="MAE Comparison (Lower is better)",
                        color='MAE',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Load and display plots if available
            st.markdown("### 📊 Detailed Analysis")
            
            results_files = [f for f in os.listdir('results') if f.endswith('.png')]
            if results_files:
                selected_plot = st.selectbox("Chọn biểu đồ:", results_files)
                st.image(f'results/{selected_plot}', use_column_width=True)
        else:
            st.warning("Chưa có kết quả evaluation. Hãy chạy training pipeline trước.")
    else:
        st.warning("Chưa có models được train. Hãy chạy training pipeline trước.")

# ==================== RUN APP ====================

if __name__ == "__main__":
    main()
