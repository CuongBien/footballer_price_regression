# Footballer Value Prediction - Streamlit App

## 📋 Cài đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements_streamlit.txt
```

Hoặc cài từng package:
```bash
pip install streamlit plotly pandas numpy scikit-learn joblib
```

### 2. Đảm bảo đã train models
Trước khi chạy Streamlit app, cần train models trước:
```bash
python main_pipeline.py
```

## 🚀 Chạy ứng dụng

### Cách 1: Chạy trực tiếp
```bash
streamlit run streamlit_app.py
```

### Cách 2: Chạy với port cụ thể
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Cách 3: Chạy trong môi trường ảo
```bash
# Windows
.venv\Scripts\activate
streamlit run streamlit_app.py

# Linux/Mac
source .venv/bin/activate
streamlit run streamlit_app.py
```

## 🎯 Tính năng chính

### 1. 📊 Batch Prediction
- Upload file CSV chứa thông tin nhiều cầu thủ
- Dự đoán giá trị hàng loạt
- Download kết quả dưới dạng CSV
- Xem biểu đồ phân phối

**Format file CSV cần thiết:**
```csv
Name,Age,Overall,Potential,Height_cm,Weight_kg,Crossing,Finishing,...
Messi,35,91,91,170,72,85,94,...
Ronaldo,38,90,90,187,84,82,93,...
```

### 2. ✏️ Manual Input
- Nhập thông tin cầu thủ thủ công
- Dự đoán với tất cả models
- So sánh kết quả từ các models khác nhau
- Biểu đồ trực quan

**Thông tin cần nhập:**
- Thông tin cơ bản: Tên, tuổi, chiều cao, cân nặng, overall, potential
- Kỹ năng tấn công: Crossing, Finishing, Dribbling, Short Passing, v.v.
- Kỹ năng phòng thủ: Interceptions, Standing Tackle, Aggression, v.v.

### 3. 📈 Model Comparison
- Xem performance của tất cả models
- So sánh R², MAE, RMSE, MAPE
- Xem các biểu đồ phân tích chi tiết
- Chọn model tốt nhất cho dự đoán

## 📂 Cấu trúc dữ liệu

App cần các file sau:
```
Footballer/
├── streamlit_app.py          # Main app
├── inference_pipeline.py      # Inference logic
├── models/                    # Trained models
│   ├── CustomRandomForestRegressor.pkl
│   ├── CustomRegressionTree_MSE.pkl
│   ├── CustomRegressionTree_MAE.pkl
│   ├── HistGradientBoosting.pkl
│   ├── KNN.pkl
│   ├── training_metadata.pkl
│   ├── training_scores.pkl
│   └── preprocessors/         # Preprocessors
├── results/                   # Evaluation results
│   ├── evaluation_report.csv
│   └── *.png                  # Charts
└── sofifa_players.csv         # Sample data
```

## 🎨 Giao diện

App có 3 tab chính:
1. **Batch Prediction**: Dự đoán hàng loạt từ file CSV
2. **Manual Input**: Nhập thông tin và dự đoán
3. **Model Comparison**: So sánh performance các models

## 🔧 Troubleshooting

### Lỗi: Module not found
```bash
pip install -r requirements_streamlit.txt
```

### Lỗi: Model file not found
- Chạy training pipeline trước: `python main_pipeline.py`

### Lỗi: Cannot load preprocessors
- Đảm bảo folder `models/preprocessors/` tồn tại và có đầy đủ files

### App chạy chậm
- Giảm số lượng dữ liệu upload
- Sử dụng 1 model thay vì tất cả models

## 📊 Demo

Sau khi chạy `streamlit run streamlit_app.py`, app sẽ mở tại:
```
http://localhost:8501
```

## 🤝 Sử dụng nâng cao

### Deploy lên Streamlit Cloud
1. Push code lên GitHub
2. Kết nối repo với Streamlit Cloud
3. Deploy tự động

### Chạy với Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements_streamlit.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📝 Notes

- App sử dụng caching để tăng tốc độ load models
- Dữ liệu input được validate tự động
- Hỗ trợ download kết quả dưới dạng CSV
- Giao diện responsive, tự động điều chỉnh theo màn hình

## 🐛 Báo lỗi

Nếu gặp lỗi, kiểm tra:
1. Python version >= 3.8
2. Đã cài đủ dependencies
3. Đã train models
4. File paths đúng

## 📚 Tài liệu tham khảo

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)
