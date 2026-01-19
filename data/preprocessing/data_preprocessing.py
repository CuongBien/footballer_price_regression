"""
Script tiền xử lý dữ liệu cho dự án dự đoán giá cầu thủ
Author: Footballer Price Regression Team
Date: 2026-01-15
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")


class FootballerDataPreprocessor:
    """
    Class xử lý tiền xử lý dữ liệu cầu thủ bóng đá
    """

    def __init__(self, data_path):
        """
        Khởi tạo preprocessor

        Args:
            data_path: Đường dẫn đến file CSV chứa dữ liệu
        """
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []

    def load_data(self):
        """Đọc dữ liệu từ file CSV"""
        print("=" * 80)
        print("BƯỚC 1: ĐỌC DỮ LIỆU")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Đã đọc dữ liệu: {self.df.shape[0]} hàng, {self.df.shape[1]} cột")
        print(
            f"✓ Các cột: {', '.join(self.df.columns[:10])}... (và {self.df.shape[1]-10} cột khác)"
        )
        return self

    def remove_unnecessary_columns(self):
        """Loại bỏ các cột không cần thiết"""
        print("\n" + "=" * 80)
        print("BƯỚC 2: LOẠI BỎ CÁC CỘT KHÔNG CẦN THIẾT")
        print("=" * 80)

        # Cột không cần thiết
        columns_to_drop = [
            "Name",  # Tên cầu thủ không dùng cho dự đoán
            "Player_URL",  # URL không dùng
            "Value_Raw",  # Đã có Value_Numeric
            "Wage_Raw",  # Đã có Wage_Numeric
            "Unnamed: 43",  # Cột trống
            "Unnamed: 44",  # Cột trống
            "Work_Rate",  # Trùng với work_rate (cột khác)
            "Team",  # Team có thể gây overfitting
        ]

        columns_exist = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=columns_exist)
        print(f"✓ Đã loại bỏ {len(columns_exist)} cột: {', '.join(columns_exist)}")
        print(f"✓ Còn lại: {self.df.shape[1]} cột")
        return self

    def handle_missing_values(self, strategy="mean"):
        """
        Xử lý giá trị thiếu (missing values)

        Args:
            strategy: Chiến lược điền giá trị thiếu ('mean', 'median', 'mode')
        """
        print("\n" + "=" * 80)
        print("BƯỚC 3: XỬ LÝ GIÁ TRỊ THIẾU")
        print("=" * 80)

        # Kiểm tra giá trị thiếu trước
        missing_before = self.df.isnull().sum()
        print(f"✓ Số giá trị thiếu trước xử lý: {missing_before.sum()}")

        # Xử lý numerical columns
        numerical_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        if strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif strategy == "median":
            imputer = SimpleImputer(strategy="median")
        else:
            imputer = SimpleImputer(strategy="most_frequent")

        self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
        print(
            f"✓ Đã điền giá trị thiếu cho {len(numerical_cols)} cột số bằng {strategy}"
        )

        # Xử lý categorical columns
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                # Điền giá trị phổ biến nhất hoặc 'Unknown'
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col].fillna(mode_value[0], inplace=True)
                else:
                    self.df[col].fillna("Unknown", inplace=True)

        print(f"✓ Đã điền giá trị thiếu cho {len(categorical_cols)} cột phân loại")

        # Kiểm tra sau
        missing_after = self.df.isnull().sum()
        print(f"✓ Số giá trị thiếu sau xử lý: {missing_after.sum()}")
        return self

    def encode_categorical_features(self):
        """Mã hóa các đặc trưng phân loại"""
        print("\n" + "=" * 80)
        print("BƯỚC 4: MÃ HÓA CÁC ĐẶC TRƯNG PHÂN LOẠI")
        print("=" * 80)

        categorical_cols = ["Nationality", "Preferred_Foot", "positioning", "work_rate"]
        categorical_cols = [col for col in categorical_cols if col in self.df.columns]

        for col in categorical_cols:
            if self.df[col].dtype == "object":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"✓ Đã mã hóa '{col}': {len(le.classes_)} giá trị duy nhất")

        print(f"✓ Tổng cộng đã mã hóa {len(categorical_cols)} cột phân loại")
        return self

    def handle_outliers(self, method="iqr", threshold=1.5):
        """
        Xử lý outliers (giá trị ngoại lai)

        Args:
            method: Phương pháp xử lý ('iqr', 'zscore')
            threshold: Ngưỡng xác định outlier
        """
        print("\n" + "=" * 80)
        print("BƯỚC 5: XỬ LÝ OUTLIERS (GIÁ TRỊ NGOẠI LAI)")
        print("=" * 80)

        numerical_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        numerical_cols = [
            col
            for col in numerical_cols
            if col not in ["Value_Numeric", "Wage_Numeric"]
        ]

        total_outliers = 0

        if method == "iqr":
            for col in numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Đếm outliers
                outliers_count = (
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                ).sum()
                total_outliers += outliers_count

                # Cắt giá trị outlier (clipping)
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == "zscore":
            from scipy import stats

            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers_count = (z_scores > threshold).sum()
                total_outliers += outliers_count

                # Loại bỏ các hàng có z-score > threshold
                self.df = self.df[(z_scores <= threshold)]

        print(f"✓ Đã xử lý {total_outliers} outliers bằng phương pháp '{method}'")
        print(f"✓ Kích thước dữ liệu sau xử lý outliers: {self.df.shape}")
        return self

    def feature_engineering(self):
        """Tạo các đặc trưng mới (Feature Engineering)"""
        print("\n" + "=" * 80)
        print("BƯỚC 6: TẠO CÁC ĐẶC TRƯNG MỚI (FEATURE ENGINEERING)")
        print("=" * 80)

        # Tạo BMI (Body Mass Index)
        if "Height_cm" in self.df.columns and "Weight_kg" in self.df.columns:
            self.df["BMI"] = self.df["Weight_kg"] / ((self.df["Height_cm"] / 100) ** 2)
            print(f"✓ Đã tạo đặc trưng 'BMI' (Body Mass Index)")

        # Tạo Age Group
        if "Age" in self.df.columns:
            self.df["Age_Group"] = pd.cut(
                self.df["Age"], bins=[0, 20, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4]
            )  # Young, Mid, Prime, Veteran, Old
            self.df["Age_Group"] = self.df["Age_Group"].astype(int)
            print(f"✓ Đã tạo đặc trưng 'Age_Group' (Nhóm tuổi)")

        # Tạo Potential Difference
        if "Overall" in self.df.columns and "Potential" in self.df.columns:
            self.df["Potential_Diff"] = self.df["Potential"] - self.df["Overall"]
            print(f"✓ Đã tạo đặc trưng 'Potential_Diff' (Tiềm năng phát triển)")

        # Tạo Skill Average (Trung bình kỹ năng)
        skill_cols = [
            "Crossing",
            "Finishing",
            "Short_passing",
            "Dribbling",
            "Ball_control",
            "Acceleration",
            "Sprint_speed",
            "Agility",
            "Reactions",
        ]
        skill_cols = [col for col in skill_cols if col in self.df.columns]
        if skill_cols:
            self.df["Skill_Avg"] = self.df[skill_cols].mean(axis=1)
            print(
                f"✓ Đã tạo đặc trưng 'Skill_Avg' (Trung bình kỹ năng từ {len(skill_cols)} thuộc tính)"
            )

        # Tạo Physical Score (Điểm thể lực)
        physical_cols = ["Stamina", "Strength", "Sprint_speed", "Acceleration"]
        physical_cols = [col for col in physical_cols if col in self.df.columns]
        if physical_cols:
            self.df["Physical_Score"] = self.df[physical_cols].mean(axis=1)
            print(f"✓ Đã tạo đặc trưng 'Physical_Score' (Điểm thể lực)")

        print(f"✓ Tổng cộng đã tạo 5 đặc trưng mới")
        return self

    def scale_features(self, scaler_type="standard", exclude_cols=None):
        """
        Chuẩn hóa các đặc trưng

        Args:
            scaler_type: Loại scaler ('standard', 'minmax')
            exclude_cols: Các cột không cần scale (thường là target variable)
        """
        print("\n" + "=" * 80)
        print("BƯỚC 7: CHUẨN HÓA DỮ LIỆU")
        print("=" * 80)

        if exclude_cols is None:
            exclude_cols = ["Value_Numeric", "Wage_Numeric"]

        # Lấy các cột số để scale
        numerical_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]

        # Chọn scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
            print(f"✓ Sử dụng StandardScaler (mean=0, std=1)")
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
            print(f"✓ Sử dụng MinMaxScaler (min=0, max=1)")

        # Fit và transform
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        print(f"✓ Đã chuẩn hóa {len(cols_to_scale)} đặc trưng")
        print(f"✓ Các cột không được chuẩn hóa: {', '.join(exclude_cols)}")

        self.feature_columns = cols_to_scale
        return self

    def get_processed_data(self):
        """Trả về dữ liệu đã xử lý"""
        return self.df

    def save_processed_data(self, output_path):
        """
        Lưu dữ liệu đã xử lý

        Args:
            output_path: Đường dẫn file output
        """
        print("\n" + "=" * 80)
        print("BƯỚC 8: LƯU DỮ LIỆU ĐÃ XỬ LÝ")
        print("=" * 80)

        self.df.to_csv(output_path, index=False)
        print(f"✓ Đã lưu dữ liệu vào: {output_path}")
        print(
            f"✓ Kích thước cuối cùng: {self.df.shape[0]} hàng, {self.df.shape[1]} cột"
        )
        return self

    def get_summary(self):
        """In tổng quan về dữ liệu đã xử lý"""
        print("\n" + "=" * 80)
        print("TỔNG QUAN DỮ LIỆU SAU TIỀN XỬ LÝ")
        print("=" * 80)
        print(f"Kích thước: {self.df.shape}")
        print(f"\nCác kiểu dữ liệu:")
        print(self.df.dtypes.value_counts())
        print(f"\nGiá trị thiếu: {self.df.isnull().sum().sum()}")
        print(f"\nThống kê mô tả:")
        print(self.df.describe())


def preprocess_pipeline(input_path, output_path):
    """
    Pipeline tiền xử lý hoàn chỉnh

    Args:
        input_path: Đường dẫn file input
        output_path: Đường dẫn file output

    Returns:
        DataFrame đã được xử lý
    """
    preprocessor = FootballerDataPreprocessor(input_path)

    # Thực hiện các bước tiền xử lý
    preprocessor.load_data().remove_unnecessary_columns().handle_missing_values(
        strategy="median"
    ).encode_categorical_features().handle_outliers(
        method="iqr", threshold=1.5
    ).feature_engineering().scale_features(
        scaler_type="standard"
    ).save_processed_data(
        output_path
    )

    # In tổng quan
    preprocessor.get_summary()

    return preprocessor.get_processed_data()


if __name__ == "__main__":
    # Đường dẫn file
    input_file = "../raw/sofifa_players.csv"
    output_file = "../processed/sofifa_players_processed.csv"

    # Chạy pipeline
    print("\n" + "🔥" * 40)
    print("TIỀN XỬ LÝ DỮ LIỆU DỰ ÁN DỰ ĐOÁN GIÁ CẦU THỦ")
    print("🔥" * 40 + "\n")

    processed_data = preprocess_pipeline(input_file, output_file)

    print("\n" + "✅" * 40)
    print("HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
    print("✅" * 40 + "\n")
