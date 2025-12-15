# ================================
# app.py — CLEAN & STABLE VERSION
# ================================

import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# =========================================================
# PATH & DUMMY MODEL (AMAN UNTUK STREAMLIT CLOUD)
# =========================================================
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def load_stt_model(): return "STT_Model"
def load_text_models(): return None, None, None
def load_embedder_model(): return "Embedder_Model"
def video_to_wav(v, a): pass
def noise_reduction(a, b): pass
def transcribe_and_clean(a, b, c, d, e): return "Dummy transcript", 0.95
def compute_confidence_score(t, p): return 0.95
def analyze_non_verbal(a): return {
    "tempo_bpm": "135",
    "total_pause_seconds": "5.2",
    "qualitative_summary": "normal"
}
def score_with_rubric(a, b, c, d, e): return 4, "Excellent answer quality."

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SEI-AI Interviewer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# SESSION STATE
# =========================================================
st.session_state.setdefault("page", "home")
st.session_state.setdefault("current_q", 1)
st.session_state.setdefault("answers", {})
st.session_state.setdefault("results", None)

TOTAL_QUESTIONS = 5
VIDEO_MAX_SIZE_MB = 50

def next_page(p):
    st.session_state.page = p
    st.rerun()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_questions():
    return {
        "1": {"question": "Tell me about a time you handled a conflict in a team."},
        "2": {"question": "What are your strengths and weaknesses?"},
        "3": {"question": "Where do you see yourself in five years?"},
        "4": {"question": "Why do you want to work for this company?"},
        "5": {"question": "Describe a technical challenge you solved."}
    }

@st.cache_data
def load_rubric():
    return {f"q{i}": {} for i in range(1, 6)}

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric()

# =========================================================
# GLOBAL CSS (SATU SUMBER, KONSISTEN)
# =========================================================
def inject_css():
    st.markdown("""
    <style>
    header, footer, #MainMenu {visibility: hidden;}
    .stApp {padding:0;}

    /* NAVBAR */
    .navbar {
        padding:20px 60px;
        display:flex;
        justify-content:space-between;
        align-items:center;
        border-bottom:1px solid #eee;
    }
    .nav-title {font-size:20px;font-weight:700;}
    .nav-links span {margin-left:24px;font-weight:500;}

    /* HERO */
    .hero {
        text-align:center;
        padding:120px 20px 80px;
        max-width:900px;
        margin:auto;
    }
    .hero h1 {font-size:48px;}
    .hero p {font-size:18px;color:#666;}

    /* BUTTON */
    .hero-btn button {
        background:black!important;
        color:white!important;
        border-radius:40px!important;
        padding:14px 40px!important;
    }

    /* STEPS */
    .steps {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:32px;
        max-width:1100px;
        margin:auto;
    }
    .step-card {
        background:#f9f9ff;
        padding:30px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 4px 12px rgba(0,0,0,.05);
    }

    /* METRIC CARDS */
    .metric-grid {
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:24px;
    }
    .metric-card {
        background:white;
        padding:24px;
        border-radius:14px;
        box-shadow:0 4px 16px rgba(0,0,0,.08);
        text-align:center;
    }

    /* SUMMARY */
    .summary-box {
        background:#fafafa;
        padding:24px;
        border-radius:14px;
    }

    /* FOOTER */
    .footer {
        background:black;
        color:#aaa;
        padding:24px 40px;
        display:flex;
        justify-content:space-between;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================
def render_home():
    inject_css()

    st.markdown("""
    <div class="navbar">
        <div class="nav-title">SEI-AI</div>
        <div class="nav-links">
            <span>Home</span>
            <span>AI Interview</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h1>SEI-AI Interviewer</h1>
        <p>Practice interviews with AI-powered evaluation & feedback.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-btn" style="text-align:center">', unsafe_allow_html=True)
    if st.button("▶️ Start Interview"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page("interview")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;margin:80px 0 40px'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="steps">
        <div class="step-card">Upload video answer</div>
        <div class="step-card">AI analyzes speech</div>
        <div class="step-card">Semantic scoring</div>
        <div class="step-card">Feedback & insights</div>
        <div class="step-card">Improve & retry</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <div>SEI-AI Interviewer</div>
        <div>© 2024 All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# INTERVIEW PAGE (LOGIC TIDAK DIUBAH)
# =========================================================
def render_interview():
    q = st.session_state.current_q
    st.title(f"Question {q} of {TOTAL_QUESTIONS}")
    st.info(QUESTIONS[str(q)]["question"])

    file = st.file_uploader("Upload video", type=["mp4","mov","webm"])
    if file:
        st.session_state.answers[str(q)] = file
        st.video(file)

    col1, col2 = st.columns(2)
    if q > 1 and col1.button("⬅ Previous"):
        st.session_state.current_q -= 1
        st.rerun()

    if q < TOTAL_QUESTIONS:
        if col2.button("Next ➡"):
            st.session_state.current_q += 1
            st.rerun()
    else:
        if col2.button("Finish & Analyze ▶️"):
            next_page("processing")

# =========================================================
# PROCESSING & FINAL SUMMARY (DIPERTAHANKAN)
# =========================================================
def render_processing():
    st.info("Processing...")
    st.session_state.results = {"done": True}
    next_page("final")

def render_final():
    inject_css()
    st.title("Final Evaluation")

    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">🎯 Avg Score<br><b>4.0</b></div>
        <div class="metric-card">🤖 Accuracy<br><b>95%</b></div>
        <div class="metric-card">⏱ Tempo<br><b>135 BPM</b></div>
        <div class="metric-card">⏸ Pause<br><b>5.2s</b></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start New Interview"):
        st.session_state.clear()
        next_page("home")

# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "interview":
    render_interview()
elif st.session_state.page == "processing":
    render_processing()
elif st.session_state.page == "final":
    render_final()
