# =========================
# app.py
# =========================

import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# ================= PATH =================
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# ================= IMPORT ML =================
try:
    from models.stt_processor import (
        load_stt_model,
        load_text_models,
        video_to_wav,
        noise_reduction,
        transcribe_and_clean
    )
    from models.scoring_logic import (
        load_embedder_model,
        compute_confidence_score,
        score_with_rubric
    )
    from models.nonverbal_analysis import analyze_non_verbal
except ImportError as e:
    st.error(f"Gagal memuat modul ML: {e}")
    st.stop()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SEI-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_q" not in st.session_state:
    st.session_state.current_q = 1
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "results" not in st.session_state:
    st.session_state.results = None

TOTAL_QUESTIONS = 5
VIDEO_MAX_SIZE_MB = 50

# ================= UTIL =================
def next_page(page):
    st.session_state.page = page
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def get_models():
    stt = load_stt_model()
    embedder = load_embedder_model()
    try:
        spell, _, english_words = load_text_models()
    except:
        spell, english_words = None, None
    return stt, embedder, spell, english_words

STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

# ================= LOAD DATA =================
@st.cache_data
def load_questions():
    with open("questions.json") as f:
        return json.load(f)

@st.cache_data
def load_rubric():
    with open("rubric_data.json") as f:
        return json.load(f)

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric()

# ======================================================
# ================= LANDING PAGE ========================
# ======================================================

def inject_landing_css():
    st.markdown("""
    <style>
    #landing-page { width:100%; }

    #landing-page .navbar {
        position: sticky;
        top: 0;
        background: white;
        padding: 20px 60px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eee;
        z-index: 10;
    }

    #landing-page .hero {
        padding: 100px 20px;
        text-align: center;
    }

    #landing-page .hero h1 {
        font-size: 44px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    #landing-page .hero p {
        max-width: 600px;
        margin: auto;
        color: #666;
        font-size: 18px;
    }

    #landing-page .steps {
        padding: 80px 40px;
        background: #fafafa;
        text-align: center;
    }

    #landing-page .step-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        min-height: 220px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    #landing-page .step-number {
        width: 44px;
        height: 44px;
        background: black;
        color: white;
        border-radius: 50%;
        margin: 0 auto 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }

    #landing-page .footer {
        background: black;
        color: #aaa;
        padding: 25px 60px;
        display: flex;
        justify-content: space-between;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)

def render_home_page():
    inject_landing_css()
    st.markdown('<div id="landing-page">', unsafe_allow_html=True)

    # NAVBAR
    col1, col2, col3 = st.columns([6,2,2])
    with col1:
        st.markdown("### SEI-AI")
    with col2:
        if st.button("Info Aplikasi"):
            next_page("info")
    with col3:
        if st.button("Mulai Wawancara"):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page("interview")

    # HERO
    st.markdown("""
    <div class="hero">
        <h1>SEI-AI Interviewer</h1>
        <p>
            Platform simulasi wawancara berbasis AI untuk melatih
            komunikasi, kepercayaan diri, dan kesiapan kerja.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Mulai Wawancara Sekarang"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page("interview")

    # STEPS
    st.markdown('<div class="steps"><h2>Bagaimana Cara Menggunakan</h2></div>', unsafe_allow_html=True)

    cols = st.columns(5)
    steps = [
        ("1","Upload Video","Jawaban wawancara"),
        ("2","STT","Audio ke teks"),
        ("3","Analisis","AI menilai"),
        ("4","Feedback","Skor & alasan"),
        ("5","Latihan","Tingkatkan skill")
    ]

    for col,(n,t,d) in zip(cols,steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{n}</div>
                <b>{t}</b><br>
                <small>{d}</small>
            </div>
            """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        <div><b>SEI-AI Interviewer</b></div>
        <div>© 2024 All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# ================= PAGE LAIN (UTUH) ====================
# ======================================================

def render_info_page():
    st.title("Informasi Aplikasi SEI-AI")
    st.markdown("""
    Aplikasi simulasi wawancara berbasis AI menggunakan:
    - Whisper (STT)
    - Sentence Transformer
    - Analisis non-verbal audio
    """)
    if st.button("🏠 Kembali"):
        next_page("home")

def render_interview_page():
    st.title(f"Pertanyaan {st.session_state.current_q}")
    qid = str(st.session_state.current_q)
    st.info(QUESTIONS[qid]["question"])

    uploaded = st.file_uploader("Upload Video", type=["mp4","mov","webm"])
    st.session_state.answers[qid] = uploaded

    if st.button("Next"):
        if st.session_state.current_q < TOTAL_QUESTIONS:
            st.session_state.current_q += 1
        else:
            next_page("processing")

def render_processing_page():
    st.title("Processing...")
    st.info("Sedang memproses jawaban Anda...")
    next_page("results")

def render_results_page():
    st.title("Hasil Analisis")
    st.success("Analisis selesai.")
    if st.button("🏠 Kembali ke Awal"):
        st.session_state.clear()
        next_page("home")

# ================= ROUTING =================
if st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "info":
    render_info_page()
elif st.session_state.page == "interview":
    render_interview_page()
elif st.session_state.page == "processing":
    render_processing_page()
elif st.session_state.page == "results":
    render_results_page()
