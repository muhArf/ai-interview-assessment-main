# app.py
import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# ================= PATH SETUP =================
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# ================= IMPORT MODEL =================
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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SEI-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= SESSION STATE =================
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

# =================================================
# =============== LANDING PAGE =====================
# =================================================

def inject_custom_css():
    st.markdown("""
    <style>
    .block-container {padding-top:0!important;}
    #MainMenu, footer, header {visibility:hidden;}

    .lp-navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 20px 60px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eee;
    }

    .lp-nav-right {
        display: flex;
        gap: 15px;
        align-items: center;
    }

    .lp-hero {
        padding: 100px 20px;
        text-align: center;
    }

    .lp-hero h1 {
        font-size: 46px;
        font-weight: 700;
    }

    .lp-hero p {
        max-width: 650px;
        margin: 20px auto 40px;
        color: #666;
        font-size: 18px;
    }

    .lp-primary button {
        background: black !important;
        color: white !important;
        border-radius: 40px !important;
        padding: 14px 40px !important;
        font-size: 16px !important;
    }

    .lp-steps {
        padding: 80px 40px;
        background: #fafafa;
        text-align: center;
    }

    .lp-step-card {
        background: white;
        padding: 30px 20px;
        border-radius: 12px;
        min-height: 230px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }

    .lp-step-number {
        width: 48px;
        height: 48px;
        background: black;
        color: white;
        border-radius: 50%;
        margin: auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 20px;
    }

    .lp-footer {
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
    inject_custom_css()

    # NAVBAR
    st.markdown('<div class="lp-navbar">', unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])

    with col1:
        try:
            st.image("assets/logo dicoding.png", width=70)
        except:
            st.markdown("**SEI-AI**")

    with col2:
        st.markdown('<div class="lp-nav-right">', unsafe_allow_html=True)
        st.markdown("Home")
        if st.button("Info Aplikasi"):
            next_page("info")
        if st.button("Mulai Wawancara"):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page("interview")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <section class="lp-hero">
        <h1>SEI-AI Interviewer</h1>
        <p>Latihan wawancara berbasis AI untuk meningkatkan kepercayaan diri dan kualitas komunikasi Anda.</p>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lp-primary" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("▶️ Mulai Sekarang"):
        next_page("interview")
    st.markdown('</div>', unsafe_allow_html=True)

    # HOW TO USE
    st.markdown('<section class="lp-steps"><h2>Bagaimana Cara Menggunakan</h2></section>', unsafe_allow_html=True)
    cols = st.columns(5)
    steps = [
        ("1","Upload Video","Unggah jawaban wawancara"),
        ("2","STT","Audio → teks otomatis"),
        ("3","Analisis","Semantik & non-verbal"),
        ("4","Skor","Evaluasi AI"),
        ("5","Latihan","Perbaiki performa")
    ]
    for col,(n,t,d) in zip(cols,steps):
        with col:
            st.markdown(f"""
            <div class="lp-step-card">
                <div class="lp-step-number">{n}</div>
                <b>{t}</b><br>
                <small>{d}</small>
            </div>
            """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="lp-footer">
        <div><b>SEI-AI Interviewer</b></div>
        <div>© 2024 All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)

# =================================================
# =============== PAGE LAIN (ASLI) =================
# =================================================

def render_info_page():
    st.title("Informasi Aplikasi SEI-AI")
    st.write("Aplikasi simulasi wawancara berbasis AI.")
    if st.button("🏠 Kembali"):
        next_page("home")

def render_interview_page():
    st.title(f"Pertanyaan {st.session_state.current_q}")
    q = QUESTIONS[str(st.session_state.current_q)]["question"]
    st.info(q)

    uploaded = st.file_uploader("Upload Video", type=["mp4","mov","webm"])
    st.session_state.answers[str(st.session_state.current_q)] = uploaded

    if st.button("Next"):
        if st.session_state.current_q < TOTAL_QUESTIONS:
            st.session_state.current_q += 1
        else:
            next_page("processing")

def render_processing_page():
    st.title("Processing...")
    next_page("results")

def render_results_page():
    st.title("Hasil Analisis")
    st.success("Analisis selesai.")
    if st.button("🏠 Kembali"):
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
