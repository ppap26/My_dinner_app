import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time

# ตั้งค่าหน้าก่อนเรียกใช้ Streamlit อื่นๆ
st.set_page_config(page_title="แนะนำร้านอาหารมื้อเย็น", page_icon="🍽️", layout="wide")

# โหลดข้อมูล
@st.cache_data
def load_data():
    file_path = "Lineman_Shops_Final_Clean.csv"
    df = pd.read_csv(file_path)

    # สร้าง rating แบบสุ่ม
    np.random.seed(42)
    df["rating"] = np.random.uniform(3.0, 5.0, len(df))
    df["combined_features"] = df["category"] + " " + df["price_level"] + " " + df["rating"].astype(str)
    return df

df = load_data()

# สร้างโมเดล TF-IDF + Nearest Neighbors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

nn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="auto")
nn_model.fit(tfidf_matrix)

def format_url(name, url):
    if pd.isna(url) or url.strip() == "-" or url.strip() == "":
        return f"https://www.google.com/search?q={name} ร้านอาหาร"
    return url

def recommend_restaurants(category, price_level, rating, top_n=10):
    filtered_df = df[(df["category"].str.contains(category, na=False, case=False)) & 
                     (df["price_level"] == price_level) & 
                     (df["rating"] >= rating)]
    
    results = []
    for _, row in filtered_df.head(top_n).iterrows():
        paragraph = f"""
        <div style='background:#ffffff; padding:15px; border-radius:10px; margin-bottom:15px; 
                     box-shadow: 0px 0px 10px rgba(0,0,0,0.1);'>
            <b style='font-size:18px; color:#28a745;'>🍴 {row['name']}</b><br>
            <small style='color:#555;'>ประเภท: {row['category']}</small><br>
            <small style='color:#555;'>ระดับราคา: {row['price_level']}</small><br>
            <small style='color:#555;'>⭐ เรตติ้ง: {row['rating']:.1f}</small><br>
            <small style='color:#555;'>📍 ที่อยู่: {row.get('street', 'ไม่ระบุ')}</small><br>
            <small style='color:#555;'>📞 โทร: {row.get('phone', 'ไม่ระบุ')}</small><br>
            <a href='{format_url(row['name'], row['url'])}' target='_blank' style='color:#007aff; font-weight:bold;'>🔗 ดูรายละเอียดเพิ่มเติม</a>
        </div>
        """
        results.append(paragraph)
    
    return results if results else ["❌ ไม่พบร้านที่ตรงกับเงื่อนไข"]

# UI
st.markdown("<h1 style='text-align: center;'>🍽️ แนะนำร้านอาหารมื้อเย็น</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px; color: grey; text-align:center;'>⬅️ เลือกตัวเลือกด้านซ้ายแล้วกดค้นหา!</p>", unsafe_allow_html=True)
st.sidebar.header("มื้อเย็นวันนี้กินอะไรดี?")

category = st.sidebar.selectbox("🍲 เลือกประเภทอาหาร", df["category"].dropna().unique())
price_level = st.sidebar.selectbox("💵 เลือกระดับราคา", df["price_level"].unique())
rating = st.sidebar.slider("🌟 เลือกเรตติ้งขั้นต่ำ", min_value=0.0, max_value=5.0, step=0.1, value=3.5)

# แสดงอนิเมชั่นก่อนค้นหา
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False

if st.sidebar.button("🔍 ค้นหาร้านอาหาร"):
    st.session_state.search_clicked = True

if st.session_state.search_clicked:
    with st.spinner("กำลังค้นหาร้านอาหาร... ⏳"):
        time.sleep(2)  # เพิ่มหน่วงเวลาให้ดูสมจริง
    results = recommend_restaurants(category, price_level, rating)
    for res in results:
        st.markdown(res, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='display:flex; justify-content:center; align-items:center; height:400px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3075/3075977.png' width='250'>
        </div>
        """, unsafe_allow_html=True)
