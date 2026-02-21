# =====================================================
# IMPORTS
# =====================================================
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd
import pickle
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="E-Commerce Clothing Intelligence",
    layout="wide",
    page_icon="🛍️"
)

# =====================================================
# PREMIUM UI CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: glow 2s infinite;
}

@keyframes glow {
    0% { text-shadow: 0 0 5px #00f5a0; }
    50% { text-shadow: 0 0 20px #00f5a0; }
    100% { text-shadow: 0 0 5px #00f5a0; }
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 20px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    border: none;
}

.stProgress > div > div > div > div {
    background-color: #00f5a0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="main-title">🛍️ E-Commerce Clothing Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multimodal AI System (Computer Vision + NLP)</div>', unsafe_allow_html=True)
st.write("---")

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    image_model = MobileNetV2(weights="imagenet")
    sentiment_model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return image_model, sentiment_model, vectorizer

image_model, model, vectorizer = load_models()

# =====================================================
# IMAGE DETECTION FUNCTION
# =====================================================
def detect_clothing_type(uploaded_file):

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = image_model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    best_label = decoded[0][1]
    best_prob = decoded[0][2] * 100
    label = best_label.lower()

    if any(word in label for word in
           ["shirt", "tshirt", "jersey", "sweatshirt", "blouse", "button_down"]):
        category = "Shirt"
    elif any(word in label for word in
             ["dress", "gown", "abaya"]):
        category = "Dress"
    elif any(word in label for word in
             ["jean", "pant", "trouser"]):
        category = "Pants"
    elif any(word in label for word in
             ["coat", "jacket", "cardigan", "trench_coat"]):
        category = "Jacket"
    else:
        category = "Other"

    return category, round(best_prob, 2), decoded

# =====================================================
# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("reviews.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review TEXT,
    sentiment TEXT,
    category TEXT
)
""")
conn.commit()
# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.radio(
    "Navigation",
    ["🔍 Predict Sentiment", "📊 Advanced Analytics"]
)

# =====================================================
# PREDICTION PAGE
# =====================================================
if menu == "🔍 Predict Sentiment":

    st.subheader("📸 Upload Clothing Item Image")

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png"]
    )

    detected_category = "Not Detected"
    confidence = 0
    top3 = []

    if uploaded_file is not None:

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(uploaded_file, caption="Uploaded Image", width=350)

        # Animated AI Vision Loading
        with st.spinner("🧠 Initializing Vision AI..."):
            time.sleep(1)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for percent in range(0, 101, 20):
            status_text.text(f"Analyzing image... {percent}%")
            progress_bar.progress(percent)
            time.sleep(0.3)

        detected_category, confidence, top3 = detect_clothing_type(uploaded_file)

        progress_bar.empty()
        status_text.empty()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.success(f"🧥 Detected Category: {detected_category}")
        st.info(f"Confidence: {confidence}%")
        st.progress(int(confidence))

        if confidence < 50:
            st.warning("⚠ Low confidence prediction.")

        st.write("### 🔍 Top 3 Predictions")
        for i in top3:
            st.write(f"{i[1]} : {round(i[2]*100, 2)}%")

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("✍ Enter Customer Review")

    user_input = st.text_area("Type review here")

    if st.button("Analyze Sentiment"):

        if user_input.strip() == "":
            st.warning("Please enter review text.")
        else:

            # Animated NLP Processing
            with st.spinner("🧠 Running NLP Sentiment Model..."):
                time.sleep(1)

            progress_bar2 = st.progress(0)
            status_text2 = st.empty()

            for percent in range(0, 101, 25):
                status_text2.text(f"Evaluating sentiment... {percent}%")
                progress_bar2.progress(percent)
                time.sleep(0.25)

            vector_input = vectorizer.transform([user_input.lower()])
            prediction = model.predict(vector_input)[0]

            progress_bar2.empty()
            status_text2.empty()

            st.info(f"Detected Product Category: {detected_category}")

            time.sleep(0.5)

            if prediction == "Positive":
                st.success("✅ Positive Review")
            elif prediction == "Negative":
                st.error("❌ Negative Review")
            else:
                st.warning("⚠ Neutral Review")

            c.execute(
                "INSERT INTO reviews VALUES (?, ?, ?)",
                (user_input, prediction, detected_category)
            )
            conn.commit()

# =====================================================
# ANALYTICS PAGE
# =====================================================
if menu == "📊 Advanced Analytics":

    st.subheader("📊 Customer Sentiment Dashboard")

    df = pd.read_sql("SELECT * FROM reviews", conn)

    if len(df) > 0:

        total_reviews = len(df)
        positive = len(df[df["sentiment"] == "Positive"])
        negative = len(df[df["sentiment"] == "Negative"])
        neutral = len(df[df["sentiment"] == "Neutral"])

        col1, col2, col3, col4 = st.columns(4)

        for col, label, value in zip(
            [col1, col2, col3, col4],
            ["Total Reviews", "Positive", "Negative", "Neutral"],
            [total_reviews, positive, negative, neutral]
        ):
            with col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric(label, value)
                st.markdown('</div>', unsafe_allow_html=True)

        st.write("---")

        st.subheader("📊 Sentiment Distribution")

        fig1, ax1 = plt.subplots(figsize=(8,5))
        sns.countplot(
            data=df,
            x="sentiment",
            palette=["#00f5a0", "#ff4b5c", "#f9c74f"],
            ax=ax1
        )
        ax1.set_facecolor("#1c1f2b")
        fig1.patch.set_facecolor("#1c1f2b")
        st.pyplot(fig1)

        st.subheader("👕 Reviews by Clothing Category")

        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.countplot(data=df, x="category", palette="coolwarm", ax=ax2)
        ax2.set_facecolor("#1c1f2b")
        fig2.patch.set_facecolor("#1c1f2b")
        st.pyplot(fig2)

        st.subheader("☁ Word Cloud")

        text = " ".join(df["review"].astype(str))
        wordcloud = WordCloud(width=1000, height=400, background_color="black").generate(text)

        fig_wc, ax_wc = plt.subplots(figsize=(12,6))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        st.subheader("📋 Stored Reviews")
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No reviews stored yet.")

# =====================================================
# FOOTER
# =====================================================

