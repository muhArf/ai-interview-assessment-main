# app.py (VERSI LENGKAP dengan semua fungsi utuh + report profesional)

import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np
import base64
import uuid
from datetime import datetime

def get_local_time_indonesia():
    """Get current time in Indonesia timezone."""
    try:
        from datetime import datetime, timedelta
        utc_now = datetime.utcnow()
        local_time = utc_now + timedelta(hours=7)
        return local_time
    except:
        return datetime.now()

# Add current directory and 'models' to PATH
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import logic from models folder
try:
    def load_stt_model(): return "STT_Model_Loaded"
    def load_text_models(): return None, None, None
    def load_embedder_model(): return "Embedder_Model_Loaded"
    def video_to_wav(video_path, audio_path): pass
    def noise_reduction(audio_path_in, audio_path_out): pass
    def transcribe_and_clean(audio_path, stt_model, spell_checker, embedder_model, english_words): return "This is a dummy transcript for testing.", 0.95
    def compute_confidence_score(transcript, log_prob_raw): return 0.95
    def analyze_non_verbal(audio_path): return {'tempo_bpm': '140 per minute', 'total_pause_seconds': '50 seconds', 'qualitative_summary': 'Normal pace'}
    def score_with_rubric(q_key_rubric, q_text, transcript, RUBRIC_DATA, embedder_model): return 4, "Candidate meets rubric 4 because dfg, candidate appears confident, but candidate has"
    
    from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
    from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
    from models.nonverbal_analysis import analyze_non_verbal
except ImportError as e:
    st.error(f"Failed to load modules from the 'models' folder. Ensure the folder structure and files are correct. Error: {e}")
    st.stop() 

# Page Configuration & Data Load
st.set_page_config(
    page_title="SEI-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_q' not in st.session_state:
    st.session_state.current_q = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'candidate_data' not in st.session_state:
    st.session_state.candidate_data = None
if 'interview_id' not in st.session_state:
    st.session_state.interview_id = None

# Constants
TOTAL_QUESTIONS = 5
VIDEO_MAX_SIZE_MB = 50

# --- Utility Functions ---
def next_page(page_name):
    st.session_state.page = page_name
    st.rerun()

@st.cache_resource
def get_models():
    """Load all heavy models (only once)."""
    try:
        stt_model = load_stt_model()
        embedder_model = load_embedder_model()
        
        try:
            spell, _, english_words = load_text_models()
        except Exception:
            spell, english_words = None, None
            st.warning("Failed to load spell checker/word models. Text cleaning will be limited.")
            
        return stt_model, embedder_model, spell, english_words
    except Exception as e:
        st.error(f"Failed to load one of the core models. Ensure all dependencies are installed. Error: {e}")
        return None, None, None, None

# Load models early
STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

@st.cache_data
def load_questions():
    """Load questions from questions.json."""
    try:
        if not os.path.exists('questions.json'):
             return {
                 "1": {"question": "Tell me about a time you handled a conflict in a team."},
                 "2": {"question": "What are your greatest strengths and weaknesses?"},
                 "3": {"question": "Where do you see yourself in five years?"},
                 "4": {"question": "Why do you want to work for this company?"},
                 "5": {"question": "Describe a difficult technical challenge you overcame."}
             }
        
        with open('questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("questions.json file not found! Please ensure the file exists.")
        return {}

@st.cache_data
def load_rubric_data():
    """Load rubric data from rubric_data.json."""
    try:
        if not os.path.exists('rubric_data.json'):
             return {
                 "q1": {"rubric": "STAR method used, clear resolution.", "keywords": ["conflict", "resolution", "STAR"]},
                 "q2": {"rubric": "Self-awareness, actionable improvements.", "keywords": ["strengths", "weaknesses", "improvement"]},
                 "q3": {"rubric": "Ambitious and company-aligned goals.", "keywords": ["goals", "future", "career"]},
                 "q4": {"rubric": "Specific reasons, knowledge of company values.", "keywords": ["company", "values", "motivation"]},
                 "q5": {"rubric": "Clear context, technical detail, result achieved.", "keywords": ["challenge", "technical", "solution"]}
             }
        
        with open('rubric_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("rubric_data.json file not found! Please ensure the file exists.")
        return {}

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()

# --- Global CSS Injection ---
def inject_global_css():
    """Inject custom CSS for all pages."""
    st.markdown("""
    <style>
    .stApp {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
        overflow-x: hidden !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* NAVBAR */
    .navbar-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: white;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.08);
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .navbar-content {
        max-width: 1200px;
        width: 100%;
        padding: 0 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        height: 50px;
    }
    
    .logo-img {
        height: 100px;
        width: auto;
        object-fit: contain;
    }
    
    .logo-text {
        font-size: 28px;
        font-weight: 800;
        color: #000000;
        margin: 0;
        text-decoration: none;
    }
    
    .nav-buttons-container {
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    .navbar-btn {
        border-radius: 25px;
        border: 2px solid #000000;
        background: transparent;
        color: #000000;
        padding: 8px 24px;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.3s ease;
        height: 40px;
        min-width: 100px;
        cursor: pointer;
        text-align: center;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        font-family: inherit;
    }
    
    .navbar-btn:hover {
        background: #000000;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-color: #000000;
    }
    
    .navbar-btn.active {
        background: #000000;
        color: white;
    }
    
    .main-content {
        padding-top: 50px !important;
        padding-left: 40px !important;
        padding-right: 40px !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* HERO SECTION */
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 24px;
        color: #000000;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 30px;
        color: #5d5988;
        max-width: 680px;
        margin: 0 auto 48px auto;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .primary-btn {
        background: #000000 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 16px 48px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .primary-btn:hover {
        background: #333333 !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* CANDIDATE FORM */
    .candidate-form-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        margin: 40px auto;
        max-width: 800px;
        border: 1px solid #f0f0f0;
    }
    
    .candidate-form-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 30px;
        color: #000000;
        text-align: center;
    }
    
    .info-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* METRIC CARDS */
    .metric-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin: 30px 0 50px 0;
    }
    
    .metric-wrapper {
        display: flex;
        justify-content: space-between;
        width: 100%;
        max-width: 1400px;
        gap: 20px;
        flex-wrap: nowrap;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        flex: 1;
        min-width: 0;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666666;
        font-weight: 500;
        text-align: center;
    }
    
    .score-color { color: #2ecc71; }
    .accuracy-color { color: #3498db; }
    .tempo-color { color: #f39c12; }
    .pause-color { color: #e74c3c; }
    
    /* CANDIDATE BANNER */
    .candidate-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .candidate-name {
        font-size: 24px;
        font-weight: 700;
    }
    
    .candidate-id {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* STEP CARDS */
    .step-card-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-bottom: 80px;
        padding: 0 20px;
    }
    
    .step-card-wrapper {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        width: 100%;
        max-width: 1400px;
        gap: 20px;
        justify-items: center;
    }
    
    .step-card {
        background: white;
        border-radius: 20px;
        padding: 60px 20px 30px 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        position: relative;
        transition: all 0.4s ease;
        border: 1px solid #f0f0f0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        height: 320px;
        width: 100%;
        max-width: 240px;
        box-sizing: border-box;
    }
    
    .step-number {
        position: absolute;
        top: -25px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        flex-shrink: 0;
    }
    
    .step-title {
        font-size: 18px;
        font-weight: 700;
        margin: 0 0 15px 0;
        color: #000000;
        line-height: 1.3;
        text-align: center;
        width: 100%;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        word-break: break-word;
        overflow-wrap: break-word;
        padding: 0 5px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        flex-shrink: 0;
    }
    
    .step-description {
        color: #666666;
        font-size: 14px;
        line-height: 1.5;
        font-weight: 400;
        text-align: center;
        width: 100%;
        flex-grow: 1;
        display: flex;
        align-items: flex-start;
        justify-content: flex-start;
        padding: 0 5px;
        word-break: break-word;
        overflow-wrap: break-word;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
    }
    
    /* FEATURES SECTION */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        margin: 60px 0;
        padding: 0 20px;
    }
    
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 35px 25px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        transition: transform 0.3s ease;
    }
    
    .feature-icon {
        font-size: 40px;
        margin-bottom: 20px;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #000000;
    }
    
    .feature-desc {
        color: #666666;
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* FOOTER */
    .custom-footer {
        background: #000000;
        color: white;
        padding: 40px 50px;
        margin-top: 100px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer-brand {
        font-size: 24px;
        font-weight: 700;
        color: white;
    }
    
    .footer-copyright {
        font-size: 14px;
        opacity: 0.7;
        font-weight: 400;
    }
    
    /* RESPONSIVE */
    @media (max-width: 1200px) {
        .main-content {
            padding-left: 30px !important;
            padding-right: 30px !important;
        }
        .metric-wrapper { flex-wrap: wrap; }
        .metric-card { flex: 0 0 calc(50% - 10px); min-width: 250px; }
        .features-grid { grid-template-columns: repeat(2, 1fr); }
        .step-card-wrapper { grid-template-columns: repeat(4, 1fr); }
        .step-card { max-width: 220px; height: 300px; padding: 50px 15px 25px 15px; }
        .step-title { font-size: 17px; height: 45px; }
        .step-description { font-size: 13px; }
    }
    
    @media (max-width: 768px) {
        .main-content {
            padding-top: 30px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        .navbar-content { padding: 0 20px; }
        .logo-text { font-size: 24px; }
        .hero-title { font-size: 42px; }
        .hero-subtitle { font-size: 28px; padding: 0 20px; }
        .candidate-form-container { padding: 25px; margin: 20px auto; }
        .candidate-form-title { font-size: 28px; }
        .candidate-banner { flex-direction: column; align-items: flex-start; gap: 10px; }
        .step-card-wrapper { grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .step-card { max-width: 180px; height: 260px; padding: 40px 10px 20px 10px; }
        .step-title { font-size: 15px; height: 35px; }
        .step-description { font-size: 12px; }
    }
    </style>
    """, unsafe_allow_html=True)

def create_navbar_html(current_page='home'):
    """Create navbar HTML dengan logo dan tombol."""
    
    logo_exists = os.path.exists("assets/seiai.png")
    
    html_parts = []
    html_parts.append('<div class="navbar-container">')
    html_parts.append('  <div class="navbar-content">')
    html_parts.append('    <div class="navbar-brand">')
    if logo_exists:
        try:
            with open("assets/seiai.png", "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
                html_parts.append(f'      <img src="data:image/png;base64,{logo_data}" alt="SEI-AI Logo" class="logo-img">')
        except:
            html_parts.append('      <div class="logo-text">SEI-AI</div>')
    else:
        html_parts.append('      <div class="logo-text">SEI-AI</div>')
    html_parts.append('    </div>')
    
    html_parts.append('    <div class="nav-buttons-container">')
    home_active = "active" if current_page == 'home' else ""
    info_active = "active" if current_page == 'info' else ""
    html_parts.append(f'      <a href="#" class="navbar-btn {home_active}" id="nav-home"> Home</a>')
    html_parts.append(f'      <a href="#" class="navbar-btn {info_active}" id="nav-info"> Info</a>')
    html_parts.append('    </div>')
    html_parts.append('  </div>')
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)

def inject_navbar_js():
    """Inject JavaScript untuk navbar secara terpisah."""
    st.markdown("""
    <script>
    function setupNavbar() {
        const homeBtn = document.getElementById('nav-home');
        if (homeBtn) {
            homeBtn.onclick = function(e) {
                e.preventDefault();
                sessionStorage.setItem('nav_to', 'home');
                window.location.reload();
            };
        }
        
        const infoBtn = document.getElementById('nav-info');
        if (infoBtn) {
            infoBtn.onclick = function(e) {
                e.preventDefault();
                sessionStorage.setItem('nav_to', 'info');
                window.location.reload();
            };
        }
    }
    
    document.addEventListener('DOMContentLoaded', setupNavbar);
    setTimeout(setupNavbar, 100);
    </script>
    """, unsafe_allow_html=True)

def render_navbar(current_page='home'):
    """Render fixed navbar dengan logo dan tombol sepenuhnya dalam HTML."""
    
    navbar_html = create_navbar_html(current_page)
    st.markdown(navbar_html, unsafe_allow_html=True)
    inject_navbar_js()
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    try:
        if 'nav_to' in st.session_state:
            nav_to = st.session_state.nav_to
            if nav_to and nav_to in ['home', 'info']:
                next_page(nav_to)
                del st.session_state.nav_to
    except:
        pass

def close_navbar():
    """Close the navbar HTML structure."""
    st.markdown("</div>", unsafe_allow_html=True)

# --- SEMUA FUNGSI RENDER PAGE ASLI (TETAP UTUH) ---

def render_home_page():
    """Render the fixed landing page."""
    inject_global_css()
    
    render_navbar('home')
    
    # HERO SECTION
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job with comprehensive evaluation and actionable insights.</p>', unsafe_allow_html=True)
    
    if st.button("Start Interview Now", key="hero_start", type="primary"):
        next_page("candidate_form")
    
    st.markdown('</section>', unsafe_allow_html=True)
    
    # HOW IT WORKS SECTION
    st.markdown('<div class="text-center" style="margin-bottom: 40px;">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 60px; color: #000000; letter-spacing: -0.5px;">How To Use</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Container untuk step cards
    st.markdown('<div class="step-card-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-card-wrapper">', unsafe_allow_html=True)
    
    steps = [
        ("1", "Candidate Registration", "Enter your personal information before starting the interview session."),
        ("2", "Upload Answer Video", "Upload your video answer for each interview question provided by the system."),
        ("3", "AI Processing", "The AI processes video into transcript and analyzes non-verbal communication aspects."),
        ("4", "Semantic Scoring", "Your answer is compared to ideal rubric criteria for content relevance scoring."),
        ("5", "Get Instant Feedback", "Receive final score, detailed rationale, and communication analysis immediately.")
    ]
    
    cols = st.columns(5)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{num}</div>
                <h3 class="step-title">{title}</h3>
                <p class="step-description">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # KEY FEATURES SECTION
    st.markdown('<h2 style="font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 60px; color: #000000; letter-spacing: -0.5px;">Key Features</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="features-grid">', unsafe_allow_html=True)
    
    features = [
        ("🎤", "Advanced Speech-to-Text", "High-accuracy audio transcription using Whisper AI model"),
        ("📊", "Comprehensive Analysis", "Evaluate content, structure, and non-verbal aspects simultaneously"),
        ("⚡", "Real-time Feedback", "Instant evaluation results and personalized recommendations"),
        ("🎯", "Rubric-based Scoring", "Objective assessment against industry-standard interview rubrics"),
        ("📈", "Progress Tracking", "Monitor improvement across multiple practice sessions"),
        ("🔒", "Privacy Focused", "Your data is processed securely and not stored permanently")
    ]
    
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(features):
                icon, title, desc = features[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <h3 class="feature-title">{title}</h3>
                        <p class="feature-desc">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("""
    <div class="custom-footer">
        <div class="footer-brand">SEI-AI Interviewer</div>
        <div class="footer-copyright">Copyright © 2025 SEI-AI Interviewer. All Rights Reserved.</div>
    </div>
    """, unsafe_allow_html=True)
    
    close_navbar()

def render_info_page():
    """Render the information page."""
    inject_global_css()
    render_navbar('info')
    
    st.title("📚 Application Information")
    
    st.markdown("""
    ### Technology Overview
    
    This application utilizes cutting-edge Machine Learning and Natural Language Processing (NLP) technologies 
    to analyze video interview answers and provide comprehensive feedback.
    
    #### Analysis Process
    1. **Speech-to-Text (STT)**: Converts spoken responses to text using the Whisper model
    2. **Text Cleaning**: Corrects spelling errors and ambiguities in the transcript
    3. **Non-Verbal Analysis**: Analyzes speaking tempo, pauses, and vocal characteristics
    4. **Semantic Scoring**: Compares answers against ideal rubric criteria using Sentence-Transformer models
    
    #### Video Requirements
    * **Duration**: Recommended 30-90 seconds per answer
    * **Format**: MP4, MOV, or WebM
    * **Maximum Size**: 50MB
    * **Audio Quality**: Ensure clear speech with minimal background noise
    
    #### Compatibility
    * **Browser**: Latest versions of Chrome, Firefox, Safari
    * **Device**: Desktop, Tablet, Smartphone
    * **Operating Systems**: Windows, macOS, Linux, Android, iOS
    
    ### Data Security
    * All videos are processed in real-time
    * No permanent data storage
    * Local processing where possible
    
    ### Support
    For technical assistance or questions:
    * Email: support@sei-ai.com
    * Phone: +1 (555) 123-4567
    * Hours: Monday-Friday, 9:00 AM - 5:00 PM EST
    """)
    
    if st.button("🏠 Back to Home", type="primary"):
        next_page('home')
    
    close_navbar()

def render_candidate_form():
    """Render form untuk input data kandidat."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown("""
    <div class="candidate-form-container">
        <h1 class="candidate-form-title">Candidate Information</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown("Please fill in your details before starting the interview.")
            
            with st.form("candidate_form"):
                name = st.text_input("Full Name", placeholder="Enter your full name", help="Your name will appear on the interview report")
                email = st.text_input("Email Address", placeholder="Enter your email address", help="We'll send the interview report to this email")
                
                submitted = st.form_submit_button("Start Interview", type="primary", use_container_width=True)
                
                if submitted:
                    if not name.strip():
                        st.error("Please enter your name")
                        return
                    if not email.strip() or "@" not in email:
                        st.error("Please enter a valid email address")
                        return
                    
                    interview_id = str(uuid.uuid4())[:8].upper()
                    
                    try:
                        now_local = get_local_time_indonesia()
                    except Exception as e:
                        now_local = get_local_time_indonesia()
                    
                    start_time = now_local.strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.session_state.candidate_data = {
                        'id': interview_id,
                        'name': name.strip(),
                        'email': email.strip(),
                        'start_time': start_time
                    }
                    st.session_state.interview_id = interview_id
                    
                    st.session_state.answers = {}
                    st.session_state.results = None
                    st.session_state.current_q = 1
                    
                    next_page("interview")
                    st.rerun()
        
        with col2:
            st.markdown("### Information")
            st.markdown("""
            <div class="info-card">
                <strong>Why we need your information:</strong>
                <ul style="margin-top: 10px; padding-left: 20px;">
                    <li>Personalize your interview experience</li>
                    <li>Include your name in the final report</li>
                    <li>Send the report to your email</li>
                    <li>Track your interview sessions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Back to Home", use_container_width=True):
        next_page('home')
    
    close_navbar()

def render_interview_page():
    """Render the interview page dengan banner kandidat."""
    inject_global_css()
    render_navbar('interview')
    
    if st.session_state.candidate_data:
        st.markdown(f"""
        <div class="candidate-banner">
            <div>
                <div class="candidate-name">Candidate: {st.session_state.candidate_data['name']}</div>
                <div class="candidate-id">Interview ID: {st.session_state.candidate_data['id']}</div>
            </div>
            <div style="text-align: right;">
                <div>Email: {st.session_state.candidate_data['email']}</div>
                <div>Started: {st.session_state.candidate_data['start_time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.title(f"🎯 Interview Question {st.session_state.current_q} of {TOTAL_QUESTIONS}")
    
    q_num = st.session_state.current_q
    q_id_str = str(q_num) 
    
    question_data = QUESTIONS.get(q_id_str, {})
    question_text = question_data.get('question', 'Question not found.')
    
    if question_text == 'Question not found.':
        st.error("An error occurred while loading the question.")
        if st.button("🏠 Back to Home"):
            st.session_state.clear() 
            next_page('home')
        return

    st.markdown("### 📝 Question:")
    st.info(f"**{question_text}**")
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file_data = st.session_state.answers.get(q_id_str)
    
    with col_upload:
        uploaded_file = None
        
        if current_uploaded_file_data is None:
            uploaded_file = st.file_uploader(
                f"📤 Upload Video Answer for Question {q_num} (Max {VIDEO_MAX_SIZE_MB}MB)",
                type=['mp4', 'mov', 'webm'],
                key=f"uploader_{q_id_str}",
                help=f"Upload a video file up to {VIDEO_MAX_SIZE_MB}MB"
            )
            
            if uploaded_file:
                if uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
                    st.error(f"File size exceeds the {VIDEO_MAX_SIZE_MB}MB limit.")
                else:
                    st.session_state.answers[q_id_str] = uploaded_file
                    current_uploaded_file_data = uploaded_file
                    st.success("✅ File successfully uploaded!")
                    st.rerun()
        
        if current_uploaded_file_data:
            st.video(current_uploaded_file_data, format=current_uploaded_file_data.type)
            st.info(f"Video for Q{q_num}: **{current_uploaded_file_data.name}**")
            
            if st.button("🗑️ Delete Video", key=f"delete_q{q_num}", type="secondary"):
                if q_id_str in st.session_state.answers:
                    del st.session_state.answers[q_id_str]
                if f"uploader_{q_id_str}" in st.session_state:
                    del st.session_state[f"uploader_{q_id_str}"]
                st.rerun()
        else:
            st.warning("Please upload your answer video to continue.")
    
    with col_control:
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("⏭️ Next Question", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("🏁 Finish & Process", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')
        
        if q_num > 1:
            if st.button("⏮️ Previous", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()
    
    close_navbar()

def render_processing_page():
    """Render the processing page dengan informasi kandidat."""
    inject_global_css()
    render_navbar('processing')
    
    if st.session_state.candidate_data:
        st.markdown(f"""
        <div style="background: #f0f2ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <strong>Candidate:</strong> {st.session_state.candidate_data['name']}
                </div>
                <div>
                    <strong>Interview ID:</strong> {st.session_state.candidate_data['id']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.title("⚙️ Analysis Process")
    st.info("Please wait, this process may take a few minutes depending on video duration.")
    
    if st.session_state.results is not None and st.session_state.results != {}:
        next_page('final_summary') 
        return
    
    if st.session_state.results is None:
        results = {}
        progress_bar = st.progress(0, text="Starting process...")
        
        if not all([STT_MODEL, EMBEDDER_MODEL]):
            st.error("Core models failed to load. Cannot proceed with processing.")
            progress_bar.empty()
            if st.button("🏠 Back to Home"):
                st.session_state.clear()
                next_page('home')
            return
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for i in range(1, TOTAL_QUESTIONS + 1):
                    q_id_str = str(i) 
                    q_key_rubric = f'q{i}' 
                    
                    video_file = st.session_state.answers.get(q_id_str)
                    q_text = QUESTIONS.get(q_id_str, {}).get('question') 
                    
                    if video_file and q_key_rubric in RUBRIC_DATA and q_text:
                        st.markdown(f"### Processing Q{i}: {q_text[:50]}...")
                        
                        progress_bar.progress((i-1)*10 + 1, text=f"Q{i}: Saving video...")
                        temp_video_path = os.path.join(temp_dir, f'video_{q_key_rubric}.mp4')
                        temp_audio_path = os.path.join(temp_dir, f'audio_{q_key_rubric}.wav')
                        
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_file.getbuffer())
                        
                        progress_bar.progress((i-1)*10 + 3, text=f"Q{i}: Extracting audio...")
                        video_to_wav(temp_video_path, temp_audio_path)
                        noise_reduction(temp_audio_path, temp_audio_path)
                        
                        progress_bar.progress((i-1)*10 + 5, text=f"Q{i}: Transcription...")
                        transcript, log_prob_raw = transcribe_and_clean(
                            temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        final_confidence_score = compute_confidence_score(transcript, log_prob_raw)
                        
                        progress_bar.progress((i-1)*10 + 7, text=f"Q{i}: Non-verbal analysis...")
                        non_verbal_res = analyze_non_verbal(temp_audio_path)
                        
                        progress_bar.progress((i-1)*10 + 9, text=f"Q{i}: Semantic scoring...")
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[ERROR: Failed to calculate score] {reason}"
                        
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value,
                            "rubric_reason": reason,
                            "confidence_score": f"{final_confidence_score*100:.2f}%",
                            "non_verbal": non_verbal_res
                        }
                        
                        progress_bar.progress(i*10, text=f"Q{i} Complete.")
                    else:
                        st.warning(f"Skipping Q{i}: Answer file not uploaded or rubric data missing.")
                
                st.session_state.results = results
                progress_bar.progress(100, text="Process complete! Redirecting to final report...")
                next_page('final_summary')
        
        except Exception as e:
            st.error(f"Fatal error during processing: {e}")
            st.warning("Processing cancelled. Please try returning to the start.")
            progress_bar.empty()
            st.session_state.results = None
            if st.button("🏠 Back to Home"):
                st.session_state.clear()
                next_page('home')
            return
    
    close_navbar()

# --- FUNGSI BARU: GENERATE PROFESSIONAL REPORT ---
def generate_professional_report(candidate_data, results, metrics):
    """Generate professional report dengan desain elegan."""
    from datetime import datetime
    
    report_date = datetime.now().strftime("%d %B %Y")
    interview_date = candidate_data.get('start_time', 'N/A')
    avg_score = metrics.get('avg_score', 0)
    
    # Performance summary text
    if avg_score >= 3.5:
        performance_summary = "Excellent performance with strong content relevance and structured responses."
    elif avg_score >= 2.5:
        performance_summary = "Good performance with adequate content. Some areas for improvement."
    else:
        performance_summary = "Performance requires improvement. Focus on content development."
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Interview Report - {candidate_data.get('name', 'Candidate')}</title>
        <style>
            @page {{
                size: A4;
                margin: 20mm;
            }}
            
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
                background: white;
            }}
            
            .report {{
                max-width: 210mm;
                margin: 0 auto;
                padding: 25mm;
            }}
            
            .header {{
                border-bottom: 3px solid #2c5282;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            
            .company-info h1 {{
                color: #2c5282;
                margin: 0;
                font-size: 32px;
                font-weight: 700;
            }}
            
            .company-info p {{
                color: #718096;
                margin: 5px 0 0 0;
                font-size: 14px;
            }}
            
            .report-info {{
                text-align: right;
                margin-top: 10px;
            }}
            
            .report-title {{
                font-size: 24px;
                font-weight: 600;
                color: #2d3748;
                margin: 0;
            }}
            
            .report-meta {{
                font-size: 12px;
                color: #718096;
                margin: 5px 0;
            }}
            
            .candidate-section {{
                background: #f7fafc;
                padding: 25px;
                border-radius: 8px;
                margin: 25px 0;
                border-left: 4px solid #4299e1;
            }}
            
            .section-title {{
                font-size: 18px;
                font-weight: 600;
                color: #2d3748;
                margin: 30px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #e2e8f0;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 15px 0;
            }}
            
            .info-item strong {{
                display: block;
                color: #4a5568;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}
            
            .info-item span {{
                font-size: 16px;
                color: #2d3748;
                font-weight: 500;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin: 25px 0;
            }}
            
            .metric-box {{
                text-align: center;
                padding: 20px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: white;
            }}
            
            .metric-value {{
                font-size: 28px;
                font-weight: 700;
                color: #2d3748;
                margin-bottom: 5px;
            }}
            
            .metric-label {{
                font-size: 12px;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .question-card {{
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
                background: white;
            }}
            
            .question-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #e2e8f0;
            }}
            
            .question-title {{
                font-size: 16px;
                font-weight: 600;
                color: #2d3748;
            }}
            
            .question-score {{
                background: #2c5282;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
            }}
            
            .question-text {{
                font-style: italic;
                color: #4a5568;
                margin: 15px 0;
                padding: 15px;
                background: #f7fafc;
                border-radius: 6px;
                border-left: 3px solid #4299e1;
            }}
            
            .question-metrics {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                margin: 20px 0;
            }}
            
            .qm-item {{
                text-align: center;
                padding: 12px;
                background: #f7fafc;
                border-radius: 6px;
            }}
            
            .qm-label {{
                font-size: 11px;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}
            
            .qm-value {{
                font-size: 16px;
                font-weight: 600;
                color: #2d3748;
            }}
            
            .evaluation {{
                background: #ebf8ff;
                padding: 20px;
                border-radius: 6px;
                margin: 20px 0;
                border-left: 3px solid #4299e1;
            }}
            
            .transcript {{
                background: #f7fafc;
                padding: 20px;
                border-radius: 6px;
                margin-top: 15px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
                max-height: 120px;
                overflow-y: auto;
            }}
            
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #e2e8f0;
                text-align: center;
                color: #718096;
                font-size: 12px;
            }}
            
            .print-button {{
                display: none;
            }}
            
            @media print {{
                .print-button {{
                    display: none !important;
                }}
                body {{
                    padding: 0 !important;
                }}
            }}
            
            @media screen {{
                .print-button {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: #2c5282;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    z-index: 1000;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report">
            <!-- HEADER -->
            <div class="header">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="company-info">
                        <h1>SEI-AI INTERVIEW ASSESSMENT</h1>
                        <p>AI-Powered Candidate Evaluation System</p>
                    </div>
                    <div class="report-info">
                        <div class="report-title">ASSESSMENT REPORT</div>
                        <div class="report-meta">Report ID: {candidate_data.get('id', 'N/A')}</div>
                        <div class="report-meta">Date: {report_date}</div>
                    </div>
                </div>
            </div>
            
            <!-- CANDIDATE INFO -->
            <div class="candidate-section">
                <div class="section-title">Candidate Information</div>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Full Name</strong>
                        <span>{candidate_data.get('name', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <strong>Interview ID</strong>
                        <span>{candidate_data.get('id', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <strong>Email Address</strong>
                        <span>{candidate_data.get('email', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <strong>Assessment Date</strong>
                        <span>{interview_date}</span>
                    </div>
                </div>
            </div>
            
            <!-- PERFORMANCE OVERVIEW -->
            <div class="section-title">Performance Overview</div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value">{metrics.get('avg_score', 0):.2f}/4</div>
                    <div class="metric-label">Average Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{metrics.get('avg_confidence', 0):.1f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{metrics.get('avg_tempo', 0):.0f} BPM</div>
                    <div class="metric-label">Speaking Tempo</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{metrics.get('total_pause', 0):.1f}s</div>
                    <div class="metric-label">Total Pauses</div>
                </div>
            </div>
            
            <div style="background: #f0fff4; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #38a169;">
                <strong>Overall Assessment:</strong> {performance_summary}
            </div>
            
            <!-- QUESTION ANALYSIS -->
            <div class="section-title">Detailed Question Analysis</div>
    """
    
    for q_key, result in results.items():
        q_num = q_key.replace('q', '')
        
        # Extract tempo and pause
        non_verbal = result.get('non_verbal', {})
        tempo_str = non_verbal.get('tempo_bpm', '0')
        pause_str = non_verbal.get('total_pause_seconds', '0')
        
        # Clean tempo
        if 'per minute' in str(tempo_str):
            tempo_value = str(tempo_str).split(' ')[0]
        else:
            tempo_value = str(tempo_str).split(' ')[0] if str(tempo_str).split(' ') else '0'
        
        # Clean pause
        if 'seconds' in str(pause_str):
            pause_value = str(pause_str).split(' ')[0]
        else:
            pause_value = str(pause_str).split(' ')[0] if str(pause_str).split(' ') else '0'
        
        html += f"""
            <div class="question-card">
                <div class="question-header">
                    <div class="question-title">Question {q_num}</div>
                    <div class="question-score">{result.get('final_score', 0)}/4</div>
                </div>
                
                <div class="question-text">{result.get('question', 'N/A')}</div>
                
                <div class="question-metrics">
                    <div class="qm-item">
                        <div class="qm-label">Confidence</div>
                        <div class="qm-value">{result.get('confidence_score', '0%')}</div>
                    </div>
                    <div class="qm-item">
                        <div class="qm-label">Tempo</div>
                        <div class="qm-value">{tempo_value} BPM</div>
                    </div>
                    <div class="qm-item">
                        <div class="qm-label">Pauses</div>
                        <div class="qm-value">{pause_value}s</div>
                    </div>
                    <div class="qm-item">
                        <div class="qm-label">Delivery</div>
                        <div class="qm-value">{non_verbal.get('qualitative_summary', 'Normal')}</div>
                    </div>
                </div>
                
                <div class="evaluation">
                    <strong>Evaluation:</strong><br>
                    {result.get('rubric_reason', 'No evaluation available.')}
                </div>
                
                <div>
                    <strong>Transcript:</strong>
                    <div class="transcript">{result.get('transcript', 'No transcript available.')}</div>
                </div>
            </div>
        """
    
    html += f"""
            <!-- FOOTER -->
            <div class="footer">
                <div>Generated by SEI-AI Interview Assessment System</div>
                <div style="margin-top: 10px; font-style: italic;">
                    Report generated on {datetime.now().strftime("%d %B %Y at %H:%M")}
                </div>
                <div style="margin-top: 5px; color: #e53e3e; font-weight: 600;">
                    CONFIDENTIAL - For assessment purposes only
                </div>
            </div>
        </div>
        
        <button class="print-button" onclick="window.print()">🖨️ Print / Save as PDF</button>
        
        <script>
            // Auto print after page loads
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    window.print();
                }}, 1000);
            }});
        </script>
    </body>
    </html>
    """
    
    return html.encode('utf-8')

def render_final_summary_page():
    """Render the final results page dengan data kandidat."""
    inject_global_css()
    render_navbar('final_summary')
    
    # Header dengan informasi kandidat
    if st.session_state.candidate_data:
        candidate = st.session_state.candidate_data
        st.markdown(f"""
        <div class="candidate-banner">
            <div>
                <div class="candidate-name">Candidate: {candidate['name']}</div>
                <div class="candidate-id">Interview ID: {candidate['id']}</div>
            </div>
            <div style="text-align: right;">
                <div>Email: {candidate['email']}</div>
                <div>Started: {candidate['start_time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.results:
        st.error("Result data not found.")
        if st.button("Back to Home"):
            next_page('home')
        return
    
    # Calculate metrics
    try:
        all_scores = [int(res['final_score']) for res in st.session_state.results.values()]
        all_confidence = [float(res['confidence_score'].replace('%', '')) 
                         for res in st.session_state.results.values()]
        
        # Extract tempo and pause
        all_tempo = []
        all_pause = []
        for res in st.session_state.results.values():
            non_verbal = res['non_verbal']
            
            # Extract tempo
            tempo_str = non_verbal.get('tempo_bpm', '0')
            if 'per minute' in str(tempo_str):
                tempo_value = str(tempo_str).split(' ')[0]
            else:
                tempo_value = str(tempo_str).split(' ')[0] if str(tempo_str).split(' ') else '0'
            
            # Extract pause
            pause_str = non_verbal.get('total_pause_seconds', '0')
            if 'seconds' in str(pause_str):
                pause_value = str(pause_str).split(' ')[0]
            else:
                pause_value = str(pause_str).split(' ')[0] if str(pause_str).split(' ') else '0'
            
            try:
                all_tempo.append(float(tempo_value))
            except ValueError:
                all_tempo.append(0)
            
            try:
                all_pause.append(float(pause_value))
            except ValueError:
                all_pause.append(0)
        
        avg_score = np.mean(all_scores) if all_scores else 0
        avg_confidence = np.mean(all_confidence) if all_confidence else 0
        avg_tempo = np.mean(all_tempo) if all_tempo else 0
        total_pause = np.sum(all_pause)
    
    except Exception as e:
        st.error(f"Failed to calculate metrics: {e}")
        return
    
    # Display metrics
    st.subheader("📊 Performance Summary")
    
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-wrapper">', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value score-color">{avg_score:.2f}/4</div>
            <div class="metric-label">Average Content Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value accuracy-color">{avg_confidence:.2f}%</div>
            <div class="metric-label">Transcript Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value tempo-color">{avg_tempo:.1f} BPM</div>
            <div class="metric-label">Average Tempo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value pause-color">{total_pause:.1f}s</div>
            <div class="metric-label">Total Pause Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Evaluation and Recommendations
    st.markdown("---")
    st.subheader("💡 Objective Evaluation & Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Conclusion")
        if avg_score >= 3.5:
            st.success("**Excellent Performance!** Content and relevance of answers were strong and well-structured.")
        elif avg_score >= 2.5:
            st.warning("**Good Performance.** Answer content was adequate but could be improved for deeper understanding.")
        else:
            st.error("**Needs Improvement.** Focus is needed on aligning answers with rubric criteria.")
        
        if 125 <= avg_tempo <= 150:
            st.success("**Optimal speaking tempo** for effective communication.")
        elif avg_tempo > 150:
            st.warning("**Speaking tempo too fast.** Practice slowing down for clarity.")
        else:
            st.warning("**Speaking tempo too slow.** Increase pace to maintain engagement.")
    
    with col2:
        st.markdown("### Key Development Areas")
        
        if avg_score < 3.0:
            st.info(
                "🎯 **Content Development:**\n"
                "- Utilize STAR method (Situation, Task, Action, Result)\n"
                "- Include specific examples and supporting data\n"
                "- Align responses with scoring rubrics"
            )
        
        if total_pause > 120:
            st.info(
                "⏸️ **Pause Management:**\n"
                "- Reduce excessively long pauses\n"
                "- Use 2-3 second pauses for emphasis only\n"
                "- Practice consistent speaking rhythm"
            )
        
        if avg_confidence < 90:
            st.info(
                "🎤 **Vocal Clarity:**\n"
                "- Increase volume and articulation\n"
                "- Choose quiet recording environments\n"
                "- Consider using external microphone"
            )
    
    # Detailed question breakdown
    st.markdown("---")
    st.subheader("📋 Detailed Breakdown by Question")
    
    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        
        st.markdown(f"### Question {q_num}")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Rubric Score", f"{res['final_score']}/4")
        
        with col_b:
            st.metric("Confidence", f"{res['confidence_score']}")
        
        with col_c:
            non_verbal = res['non_verbal']
            tempo_str = non_verbal.get('tempo_bpm', '0')
            if 'per minute' in str(tempo_str):
                tempo_display = str(tempo_str).split(' ')[0] + " BPM"
            else:
                tempo_display = str(tempo_str).split(' ')[0] + " BPM" if str(tempo_str).split(' ') else "0 BPM"
            st.metric("Tempo", tempo_display)
        
        with col_d:
            pause_str = non_verbal.get('total_pause_seconds', '0')
            if 'seconds' in str(pause_str):
                pause_display = str(pause_str).split(' ')[0] + " seconds"
            else:
                pause_display = str(pause_str).split(' ')[0] + " seconds" if str(pause_str).split(' ') else "0 seconds"
            st.metric("Pause", pause_display)
        
        st.markdown(f"**Question:** {res['question']}")
        
        st.markdown("**Evaluation:**")
        st.info(res['rubric_reason'])
        
        with st.expander("View Transcript"):
            st.code(res['transcript'], language='text')
        
        st.markdown("---")
    
    # Action buttons - DENGAN PROFESSIONAL REPORT
    st.markdown("---")
    st.subheader("📄 Download Report")
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("🔄 New Interview", use_container_width=True, type="primary"):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page('candidate_form')
    
    with col_btn2:
        if st.button("📥 JSON Report", use_container_width=True):
            if st.session_state.candidate_data:
                report_data = {
                    'candidate': st.session_state.candidate_data,
                    'results': st.session_state.results,
                    'metrics': {
                        'avg_score': avg_score,
                        'avg_confidence': avg_confidence,
                        'avg_tempo': avg_tempo,
                        'total_pause': total_pause
                    }
                }
                json_report = json.dumps(report_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON Report",
                    data=json_report,
                    file_name=f"interview_report_{st.session_state.candidate_data['id']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("Download feature requires candidate information.")
    
    with col_btn3:
        # PROFESSIONAL REPORT BUTTON
        if st.button("📄 Professional Report", use_container_width=True, type="primary"):
            if st.session_state.candidate_data:
                metrics = {
                    'avg_score': avg_score,
                    'avg_confidence': avg_confidence,
                    'avg_tempo': avg_tempo,
                    'total_pause': total_pause
                }
                
                with st.spinner("Generating professional report..."):
                    html_bytes = generate_professional_report(
                        st.session_state.candidate_data,
                        st.session_state.results,
                        metrics
                    )
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download Report (HTML)",
                        data=html_bytes,
                        file_name=f"Interview_Report_{st.session_state.candidate_data['name'].replace(' ', '_')}.html",
                        mime="text/html",
                        use_container_width=True,
                        help="Open in browser and press Ctrl+P to print/save as PDF"
                    )
                    
                    # Auto-open in new tab for printing
                    html_content = html_bytes.decode('utf-8').replace('`', '\\`').replace('${', '\\${')
                    js_code = f"""
                    <script>
                    function openReport() {{
                        var win = window.open('', '_blank');
                        win.document.write(`{html_content}`);
                        win.document.close();
                        
                        // Auto print after 1 second
                        setTimeout(function() {{
                            win.print();
                        }}, 1000);
                    }}
                    
                    // Try to open report
                    try {{
                        openReport();
                    }} catch(e) {{
                        console.log('Popup blocked, showing download button instead');
                    }}
                    </script>
                    """
                    
                    st.components.v1.html(js_code, height=0)
                    
                    st.success("✅ Professional report generated!")
                    st.info("💡 The report will open in a new window for printing. If popup is blocked, use the download button above.")
            else:
                st.error("Candidate information required to generate report.")
    
    with col_btn4:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.clear()
            next_page('home')
    
    close_navbar()

# Main App Execution Flow
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'info':
    render_info_page()
elif st.session_state.page == 'candidate_form':
    render_candidate_form()
elif st.session_state.page == 'interview':
    render_interview_page()
elif st.session_state.page == 'processing':
    render_processing_page()
elif st.session_state.page == 'final_summary':
    render_final_summary_page()