# app.py
import streamlit as st
import torch
import joblib
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# ============================================
# Konfigurasi Streamlit
# ============================================
st.set_page_config(page_title="Emotion Mining — Tom Lembong", layout="wide")

st.title("💬 Emotion Mining App — Komentar Publik tentang Tom Lembong")

st.markdown("""
**Latar Belakang Kasus:**  
Tom Lembong, mantan Menteri Perdagangan, pada tahun 2024 divonis kasus korupsi impor gula.  
Namun tak lama setelah itu, Presiden memberikan abolisi yang membatalkan proses pidana.  
Kasus ini memicu beragam reaksi publik di media sosial, terutama Instagram dan X.  

Model ini akan mengklasifikasikan komentar ke dalam 5 kategori emosi utama:
- 😢 **SADNESS**
- 😡 **ANGER**
- 🌱 **HOPE**
- 😞 **DISAPPOINTMENT**
- 🤝 **SUPPORT**
""")

# ============================================
# Load model & tokenizer (cached)
# ============================================
@st.cache_resource
def load_all():
    BASE_MODEL = "indolem/indobertweet-base-uncased"
    REPO_ID = "Oviorlanda/5emosi"

    # Load tokenizer & BERT
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bert_model = AutoModel.from_pretrained(BASE_MODEL)
    bert_model.eval()

    # Load Logistic Regression classifier
    logreg_path = hf_hub_download(repo_id=REPO_ID, filename="logreg_model.pkl")
    logreg = joblib.load(logreg_path)

    # Load id2label mapping
    id2label_path = hf_hub_download(repo_id=REPO_ID, filename="id2label.json")
    with open(id2label_path, "r") as f:
        id2label = json.load(f)

    return tokenizer, bert_model, logreg, id2label

tokenizer, bert_model, logreg_model, id2label = load_all()

# Warna tiap emosi
emotion_colors = {
    "SADNESS": "#1f77b4",
    "ANGER": "#ff7f0e",
    "HOPE": "#2ca02c",
    "DISAPPOINTMENT": "#d62728",
    "SUPPORT": "#9467bd",
}

# ============================================
# Fungsi embedding
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

@st.cache_data
def get_embedding(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ============================================
# Input & Prediksi
# ============================================
user_input = st.text_area("📝 Masukkan komentar publik (Bahasa Indonesia):", "", height=150)

if st.button("🔮 Prediksi Emosi"):
    if user_input.strip():
        emb = get_embedding(user_input)
        probs = logreg_model.predict_proba(emb)[0]
        pred = int(np.argmax(probs))
        label = id2label[str(pred)]

        # Tampilkan hasil dengan highlight warna
        st.subheader("Hasil Prediksi")
        st.markdown(
            f"<div style='background-color:{emotion_colors[label]}; "
            f"padding:20px; border-radius:10px; color:white; text-align:center;'>"
            f"<h2>{label}</h2></div>",
            unsafe_allow_html=True
        )

        # Probabilitas semua kelas
        st.write("📊 **Probabilitas untuk semua emosi:**")
        for i, p in enumerate(probs):
            lbl = id2label[str(i)]
            st.write(f"- {lbl}: {p:.2f}")
    else:
        st.warning("⚠️ Tolong masukkan komentar dulu!")
