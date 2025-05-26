
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="VP-AI: Dự đoán & Gợi ý Kháng sinh", layout="wide")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def load_and_train():
    def convert_binary(val):
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ["x", "có", "yes"]:
                return 1
            elif val in ["/", "khong", "không", "no", "ko"]:
                return 0
        return val

    def convert_age(val):
        if isinstance(val, str):
            val = val.strip()
            if "thg" in val.lower():
                return float(val.replace("Thg", "").replace("thg", "")) / 10
            try:
                return float(val)
            except:
                return np.nan
        return val

    def convert_numeric(val):
        try:
            return float(str(val).strip())
        except:
            return np.nan

    df = pd.read_csv("Mô hình AI.csv")
    df = df.rename(columns=lambda x: x.strip())  # bỏ khoảng trắng
    df.drop(columns=["So ngay dieu tri"], errors="ignore", inplace=True)

    if "Benh ngay thu truoc khi nhap vien" in df.columns:
        df.rename(columns={"Benh ngay thu truoc khi nhap vien": "Benh ngay thu"}, inplace=True)
    if " SpO2" in df.columns:
        df.rename(columns={" SpO2": "SpO2"}, inplace=True)

    df["Tuoi"] = df["Tuoi"].apply(convert_age)
    df["Benh ngay thu"] = df["Benh ngay thu"].apply(convert_numeric)
    df["SpO2"] = df["SpO2"].apply(convert_numeric)
    df = df[df["Tac nhan"].notna()]

    binary_cols = df.select_dtypes(include="object").columns.difference(["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"])
    df[binary_cols] = df[binary_cols].applymap(convert_binary)

    # Xử lý giá trị lỗi
    df = df.applymap(lambda x: x if isinstance(x, (int, float)) or pd.isnull(x) else np.nan)
    df = df.dropna(subset=df.columns.difference(["Tac nhan"]), how="any")

    X = df.drop(columns=["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"], errors="ignore")
    y = df["Tac nhan"]
    X = X.fillna(X.mean(numeric_only=True))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_cols = load_and_train()

# Gợi ý kháng sinh theo từng tác nhân gây bệnh
pathogen_to_antibiotics = {
    "H. influenzae": ["Amoxicilin clavulanic", "Ceftriaxone"],
    "K. pneumoniae": ["Meropenem", "Ceftriaxone"],
    "M. catarrhalis": ["Amoxicilin clavulanic", "Clarithromycin"],
    "M. pneumoniae": ["Clarithromycin", "Levofloxacin"],
    "RSV": [],
    "S. aureus": ["Vancomycin", "Clindamycin"],
    "S. epidermidis": ["Vancomycin"],
    "S. mitis": ["Penicillin"],
    "S. pneumoniae": ["Ceftriaxone", "Vancomycin"],
    "unspecified": []
}

st.markdown("### 📋 Nhập dữ liệu lâm sàng")

user_input = {}
for col in feature_cols:
    if col == "Tuoi":
        user_input[col] = st.number_input("Tuổi (năm)", min_value=0.0, max_value=100.0, step=1.0)
    elif col in ["Nhiet do", "Bach cau", "CRP", "Nhip tho", "Mach", "Benh ngay thu", "SpO2"]:
        user_input[col] = st.number_input(col, value=0.0)
    else:
        user_input[col] = st.radio(f"{col}:", ["Không", "Có"], horizontal=True) == "Có"

if st.button("🔍 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if isinstance(input_df[col].iloc[0], bool):
            input_df[col] = input_df[col].astype(int)

    pred = model.predict(input_df)[0]
    st.success(f"✅ Tác nhân gây bệnh được dự đoán: **{pred}**")

    antibiotics = pathogen_to_antibiotics.get(pred, [])
    st.markdown("### 💊 Kháng sinh gợi ý:")
    if antibiotics:
        for abx in antibiotics:
            st.write(f"- **{abx}**")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")
