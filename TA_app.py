import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from datetime import datetime
from io import BytesIO

# ======= Konfigurasi Halaman =======
st.set_page_config(page_title="Analisis Sentimen Twitter", layout="wide")

# ======= Sidebar =======
st.sidebar.title("🔍 Analisis Sentimen Twitter")
mode = st.sidebar.radio("Pilih Mode Input", ["Bearer Token", "Upload CSV"])

if mode == "Bearer Token":
    bearer_token = st.sidebar.text_input("Bearer Token Twitter")
    query = st.sidebar.text_input("Kata kunci pencarian", value="")
    max_results = st.sidebar.selectbox("Jumlah data yang ingin ditampilkan", [10, 50, 100])
    fetch_button = st.sidebar.button("Ambil Data Twitter")
else:
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# ======= Load Model & Tokenizer =======
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    return tokenizer, model

tokenizer, model = load_model()

# ======= Pustaka Kata Kasar =======
kata_negatif = ['anjing', 'bangsat', 'goblok', 'tolol', 'bego', 'kampret', 'idiot','hama','jawir','jembut']

# ======= Preprocessing =======
def bersihkan_teks(teks):
    teks = re.sub(r"http\S+", "", teks)
    teks = re.sub(r"@\w+", "", teks)
    teks = re.sub(r"#", "", teks)
    teks = re.sub(r"\s+", " ", teks)
    return teks.strip()

# ======= Ambil Tweet dari API =======
def ambil_tweet(query, bearer_token, max_results):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "tweet.fields": "created_at,author_id,text",
        "expansions": "author_id",
        "user.fields": "username",
        "max_results": max_results
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error("Gagal mengambil data. Pastikan Bearer Token valid.")
        return pd.DataFrame()

    hasil = response.json()
    pengguna = {u["id"]: u["username"] for u in hasil.get("includes", {}).get("users", [])}
    data = []
    for tweet in hasil.get("data", []):
        teks = tweet["text"]
        teks_bersih = bersihkan_teks(teks)
        data.append({
            "Tanggal": tweet["created_at"],
            "Akun": pengguna.get(tweet["author_id"], "anonim"),
            "Tweet": teks_bersih
        })
    return pd.DataFrame(data)

# ======= Prediksi Sentimen =======
def prediksi_sentimen(df):
    hasil = []
    for teks in df["Tweet"]:
        if any(kasar in teks.lower() for kasar in kata_negatif):
            hasil.append("Negatif")
            continue

        inputs = tokenizer(teks, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        skor = softmax(logits.numpy()[0])
        label = skor.argmax()
        if label == 0:
            hasil.append("Negatif")
        elif label == 1:
            hasil.append("Netral")
        else:
            hasil.append("Positif")
    df["Sentimen"] = hasil
    return df

# ======= WordCloud Generator =======
def generate_wordcloud(df, sentimen=None):
    if sentimen:
        df = df[df["Sentimen"] == sentimen]
    teks = " ".join(df["Tweet"].values)
    if not teks.strip():
        return None
    wc = WordCloud(width=600, height=300, background_color='white').generate(teks)
    return wc

# ======= Plot Kata Populer =======
def plot_kata_populer(df):
    all_words = " ".join(df["Tweet"]).lower().split()
    freq = pd.Series(all_words).value_counts().head(10)
    fig, ax = plt.subplots()
    freq.plot(kind='bar', ax=ax, color="skyblue")
    ax.set_title("10 Kata Paling Sering Muncul")
    return fig

# ======= Main Area =======
st.title("📊 Analisis Sentimen Twitter/X")

def buat_download(dataframe, nama_file):
    buffer = BytesIO()
    dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(f"⬇️ Download {nama_file}", data=buffer, file_name=nama_file, mime="text/csv")

if mode == "Bearer Token" and fetch_button:
    data = ambil_tweet(query, bearer_token, max_results)
    if not data.empty:
        data_asli = data.copy()
        data = prediksi_sentimen(data)
        st.subheader(f"Hasil Analisis Tweet '{query}'")
        st.dataframe(data)

        # Grafik Sentimen
        st.subheader("Grafik Sentimen")
        sentimen_count = data["Sentimen"].value_counts()
        fig1, ax1 = plt.subplots()
        sentimen_count.plot(kind="bar", ax=ax1, color=["red", "gray", "green"])
        ax1.set_ylabel("Jumlah")
        ax1.set_title("Distribusi Sentimen")
        st.pyplot(fig1)

        # Pie Chart
        st.subheader("Pie Chart Sentimen")
        fig2, ax2 = plt.subplots()
        ax2.pie(sentimen_count, labels=sentimen_count.index, autopct="%1.1f%%", colors=["red", "gray", "green"])
        ax2.axis("equal")
        st.pyplot(fig2)

        # Wordcloud
        st.subheader("Wordcloud Berdasarkan Sentimen")
        col1, col2 = st.columns(2)
        with col1:
            for label in ["Positif", "Negatif"]:
                st.markdown(f"**{label}**")
                wc = generate_wordcloud(data, label)
                if wc:
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
        with col2:
            st.markdown("**Netral**")
            wc_neutral = generate_wordcloud(data, "Netral")
            if wc_neutral:
                fig, ax = plt.subplots()
                ax.imshow(wc_neutral, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            st.markdown("**Gabungan**")
            wc_all = generate_wordcloud(data)
            if wc_all:
                fig, ax = plt.subplots()
                ax.imshow(wc_all, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        # Kata Populer
        st.subheader("🔠 Kata Paling Sering Muncul")
        st.pyplot(plot_kata_populer(data))

        # ======= Download Section =======
        st.subheader("📥 Unduh Data")
        buat_download(data, "hasil_twitter_analisis.csv")
        buat_download(data_asli, "data_asli.csv")
        buat_download(data[["Akun", "Tanggal", "Tweet"]], "data_bersih.csv")

elif mode == "Upload CSV" and uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    if "Tweet" not in df_csv.columns:
        st.error("File CSV harus memiliki kolom bernama 'Tweet'")
    else:
        df_csv_asli = df_csv.copy()
        df_csv["Tweet"] = df_csv["Tweet"].astype(str).apply(bersihkan_teks)
        df_csv = prediksi_sentimen(df_csv)
        st.subheader("Hasil Analisis dari File CSV")
        st.dataframe(df_csv)

        st.subheader("Grafik Sentimen")
        sentimen_count = df_csv["Sentimen"].value_counts()
        fig1, ax1 = plt.subplots()
        sentimen_count.plot(kind="bar", ax=ax1, color=["red", "gray", "green"])
        ax1.set_ylabel("Jumlah")
        ax1.set_title("Distribusi Sentimen")
        st.pyplot(fig1)

        # Wordcloud Gabungan
        st.subheader("Wordcloud Semua Sentimen")
        wc_csv = generate_wordcloud(df_csv)
        if wc_csv:
            fig, ax = plt.subplots()
            ax.imshow(wc_csv, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # Kata Populer
        st.subheader("🔠 Kata Paling Sering Muncul")
        st.pyplot(plot_kata_populer(df_csv))

        # ======= Download Section =======
        st.subheader("📥 Unduh Data")
        buat_download(df_csv, "hasil_analisis.csv")
        buat_download(df_csv_asli, "data_asli.csv")
        df_download = df_csv.copy()
        if "Akun" not in df_download.columns:
            df_download["Akun"] = "-"
        if "Tanggal" not in df_download.columns:
            df_download["Tanggal"] = "-"
        buat_download(df_download[["Akun", "Tanggal", "Tweet"]], "data_bersih.csv")
