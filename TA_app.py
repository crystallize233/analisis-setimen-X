import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Opini Twitter Indonesia", layout="wide")

# Muat model IndoBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")
    return tokenizer, model

tokenizer, model = load_model()
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

# Header
st.title("🇮🇩 Dashboard Analisis Sentimen Twitter/X")
st.markdown("""
Aplikasi ini menggunakan model IndoBERT untuk menganalisis sentimen tweet berbahasa Indonesia dari Twitter atau file CSV.
""")

# Sidebar
st.sidebar.header("Konfigurasi")
twitter_auth_token = st.sidebar.text_input("Twitter Bearer Token:", type="password")
mode = st.sidebar.radio("Pilih Mode Data:", ["Ambil dari Twitter", "Upload File CSV"])

# Fungsi analisis
@st.cache_data
def analisis_sentimen_indober(df):
    results = []
    for text in df['content']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = softmax(logits.numpy()[0])
        label_idx = probs.argmax()
        results.append(label_map[label_idx])
    df['label'] = results
    return df

def tampilkan_hasil(df):
    df['content'] = df['content'].astype(str)

    st.write("### Contoh komentar", df.head())

    st.write("### Distribusi Sentimen")
    st.bar_chart(df['label'].value_counts())

    st.write("### WordCloud")
    all_text = " ".join(df['content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="\U0001F4E5 Download Hasil CSV",
        data=csv,
        file_name='hasil_sentimen_indonesia.csv',
        mime='text/csv',
    )

# Mode Ambil dari Twitter
if mode == "Ambil dari Twitter":
    keyword = st.text_input("Masukkan Kata Kunci:")
    jumlah = st.slider("Jumlah Tweet:", 10, 100, 30)

    if st.button("Ambil dan Analisis Tweet"):
        if not twitter_auth_token:
            st.error("Silakan masukkan Bearer Token.")
        elif not keyword:
            st.warning("Masukkan kata kunci terlebih dahulu.")
        elif any(ord(c) > 127 for c in keyword):
            st.error("Kata kunci tidak boleh mengandung emoji atau karakter non-ASCII.")
        else:
            with st.spinner("Mengambil tweet..."):
                headers = {
                    "Authorization": f"Bearer {twitter_auth_token}",
                    "User-Agent": "StreamlitApp"
                }
                # Kata kunci + bahasa Indonesia + hanya tweet asli (bukan reply/retweet)
                query = re.sub(r'[^\w\s]', '', keyword) + " lang:id -is:retweet -is:reply"
                url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={jumlah}&tweet.fields=created_at,text,lang"

                try:
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        tweets = response.json().get("data", [])
                        if tweets:
                            df = pd.DataFrame(tweets)
                            df.rename(columns={"created_at": "date", "text": "content"}, inplace=True)
                            df = df[["date", "content"]]
                            df = analisis_sentimen_indober(df)
                            tampilkan_hasil(df)
                        else:
                            st.warning("Tidak ada tweet ditemukan untuk kata kunci tersebut.")
                    elif response.status_code == 401:
                        st.error("Token tidak valid atau tidak memiliki izin.")
                    elif response.status_code == 429:
                        st.error("Terlalu banyak permintaan. Tunggu beberapa saat dan coba lagi.")
                    else:
                        st.error(f"Gagal mengambil data. Status: {response.status_code}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

# Mode Upload CSV
elif mode == "Upload File CSV":
    uploaded_file = st.file_uploader("Unggah file CSV yang berisi kolom 'content' atau 'full_text'", type="csv")
    if uploaded_file:
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            # Deteksi kolom teks
            text_col = None
            if 'content' in df.columns:
                text_col = 'content'
            elif 'full_text' in df.columns:
                text_col = 'full_text'

            if not text_col:
                st.error("Tidak ditemukan kolom 'content' atau 'full_text' dalam file CSV.")
            else:
                df = df[[text_col]].rename(columns={text_col: 'content'})
                df = analisis_sentimen_indober(df)
                tampilkan_hasil(df)

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
