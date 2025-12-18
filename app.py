# app.py (versi dimodifikasi dengan form input kandidat)

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
        # Simple manual adjustment for Indonesia timezones
        from datetime import datetime, timedelta
        utc_now = datetime.utcnow()
        
        # Choose your timezone:
        # WIB (UTC+7): Jakarta, Sumatra, Java, West Kalimantan
        # WITA (UTC+8): Bali, Nusa Tenggara, South/East Kalimantan, Sulawesi
        # WIT (UTC+9): Maluku, Papua
        
        # Default to WIB
        local_time = utc_now + timedelta(hours=7)
        return local_time
    except:
        return datetime.now()

# Add current directory and 'models' to PATH
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import logic from models folder
try:
    # IMPORTANT: Ensure files in models/ folder are correctly implemented
    # Dummy imports if modules in models/ folder are not fully implemented
    def load_stt_model(): return "STT_Model_Loaded"
    def load_text_models(): return None, None, None
    def load_embedder_model(): return "Embedder_Model_Loaded"
    def video_to_wav(video_path, audio_path): pass
    def noise_reduction(audio_path_in, audio_path_out): pass
    def transcribe_and_clean(audio_path, stt_model, spell_checker, embedder_model, english_words): return "This is a dummy transcript for testing.", 0.95
    def compute_confidence_score(transcript, log_prob_raw): return 0.95
    def analyze_non_verbal(audio_path): return {'tempo_bpm': '135 BPM', 'total_pause_seconds': '5.2', 'qualitative_summary': 'Normal pace'}
    def score_with_rubric(q_key_rubric, q_text, transcript, RUBRIC_DATA, embedder_model): return 4, "Excellent relevance and structural clarity."
    
    # Replace with actual imports if modules exist
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
# Tambah session state untuk data kandidat
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
        # IMPORTANT: Replace model paths according to your actual implementation
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
        # Dummy data if questions.json doesn't exist
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
        # Dummy data if rubric_data.json doesn't exist
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
    /* 1. GLOBAL STREAMLIT RESET */
    .stApp {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
        overflow-x: hidden !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* 2. FIXED NAVBAR CONTAINER - FULLY IN HTML */
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
    
    /* Brand/Logo - Pure HTML */
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
    
    /* Navigation buttons container - Pure HTML */
    .nav-buttons-container {
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    /* Custom navbar button styling - Pure HTML */
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
    
    /* Active button state */
    .navbar-btn.active {
        background: #000000;
        color: white;
    }
    
    /* 3. MAIN CONTENT PADDING (to account for fixed navbar) */
    .main-content {
        padding-top: 50px !important;
        padding-left: 40px !important;
        padding-right: 40px !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* 4. LANDING PAGE HERO SECTION */
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
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
    
    /* 5. CANDIDATE FORM STYLING */
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
    
    .form-field {
        margin-bottom: 25px;
    }
    
    .form-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #333333;
        display: block;
    }
    
    .form-input {
        width: 100%;
        padding: 12px 16px;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .form-input:focus {
        outline: none;
        border-color: #000000;
        box-shadow: 0 0 0 3px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* 6. HOW IT WORKS SECTION - STEP CARDS */
    .section-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 60px;
        color: #000000;
        letter-spacing: -0.5px;
    }
    
    .step-card-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-bottom: 80px;
        padding: 0 20px;
    }
    
    .step-card-wrapper {
        display: flex;
        justify-content: space-between;
        width: 100%;
        max-width: 1400px;
        gap: 20px;
        flex-wrap: nowrap; /* Pastikan ini nowrap */
    }
    
    .step-card {
        background: white;
        border-radius: 20px;
        padding: 60px 25px 35px 25px;
        text-align: center;
        width: 250px;
        height: 320px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        position: relative;
        transition: all 0.4s ease;
        border: 1px solid #f0f0f0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        flex: 0 0 auto;
    }
    
    .step-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.12);
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
    }
    
    .step-title {
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 10px 0;
        color: #000000;
        line-height: 1.4;
        text-align: center;
        width: 100%;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .step-description {
        color: #666666;
        font-size: 15px;
        line-height: 1.6;
        font-weight: 400;
        text-align: center;
        width: 100%;
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 5px;
    }
    
    /* RESPONSIVE DESIGN UNTUK STEP CARDS - TAMBAHKAN DI SINI */
    @media (max-width: 1200px) {
        .step-card {
            width: 220px;
            padding: 50px 20px 30px 20px;
            height: 300px;
        }
    }
    
    @media (max-width: 992px) {
        .step-card {
            width: 200px;
            padding: 45px 18px 25px 18px;
            height: 280px;
        }
        
        .step-title {
            font-size: 18px;
            min-height: 50px;
        }
        
        .step-description {
            font-size: 14px;
        }
    }
    
    @media (max-width: 768px) {
        .step-card-wrapper {
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
        }
        
        .step-card {
            width: 180px;
            height: 260px;
            padding: 40px 15px 20px 15px;
            flex: 0 0 calc(50% - 15px);
        }
        
        .step-number {
            width: 45px;
            height: 45px;
            font-size: 20px;
            top: -20px;
        }
    }
    
    @media (max-width: 576px) {
        .step-card-wrapper {
            gap: 12px;
        }
        
        .step-card {
            width: 100%;
            max-width: 280px;
            flex: 0 0 100%;
            height: auto;
            min-height: 240px;
            padding: 40px 15px 25px 15px;
        }
    }
    
    /* 12. RESPONSIVE DESIGN - MEDIA QUERY YANG SUDAH ADA */
    @media (max-width: 1200px) {
        .main-content {
            padding-left: 30px !important;
            padding-right: 30px !important;
        }
        
        .metric-wrapper {
            flex-wrap: wrap;
        }
        
        .metric-card {
            flex: 0 0 calc(50% - 10px);
            min-width: 250px;
        }
        
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .main-content {
            padding-top: 30px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        
        .navbar-content {
            padding: 0 20px;
        }
        
        .logo-text {
            font-size: 24px;
        }
        
        .hero-title {
            font-size: 42px;
        }
        
        .hero-subtitle {
            font-size: 28px;
            padding: 0 20px;
        }
        
        .candidate-form-container {
            padding: 25px;
            margin: 20px auto;
        }
        
        .candidate-form-title {
            font-size: 28px;
        }
        
        .candidate-banner {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }
    }
    
    @media (max-width: 480px) {
        .candidate-form-container {
            padding: 20px;
        }
    }

    
    /* 7. FEATURES SECTION */
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
    
    .feature-card:hover {
        transform: translateY(-5px);
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
    
    /* 8. FOOTER */
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
    
    /* 9. METRIC CARDS FOR RESULTS */
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
    
    /* 10. INTERVIEW PAGE STYLING */
    .question-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 40px;
        border-left: 6px solid #667eea;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    /* 11. CANDIDATE INFO BANNER */
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
    
    /* 12. RESPONSIVE DESIGN */
    @media (max-width: 1200px) {
        .main-content {
            padding-left: 30px !important;
            padding-right: 30px !important;
        }
        
        .metric-wrapper {
            flex-wrap: wrap;
        }
        
        .metric-card {
            flex: 0 0 calc(50% - 10px);
            min-width: 250px;
        }
        
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .main-content {
            padding-top: 30px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        
        .navbar-content {
            padding: 0 20px;
        }
        
        .logo-text {
            font-size: 24px;
        }
        
        .hero-title {
            font-size: 42px;
        }
        
        .hero-subtitle {
            font-size: 28px;
            padding: 0 20px;
        }
        
        .candidate-form-container {
            padding: 25px;
            margin: 20px auto;
        }
        
        .candidate-form-title {
            font-size: 28px;
        }
        
        .candidate-banner {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }
    }
    
    @media (max-width: 480px) {
        .candidate-form-container {
            padding: 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_navbar_html(current_page='home'):
    """Create navbar HTML dengan logo dan tombol."""
    
    # Cek apakah logo file ada
    logo_exists = os.path.exists("assets/seiai.png")
    
    # Generate HTML untuk navbar
    html_parts = []
    
    # Navbar container
    html_parts.append('<div class="navbar-container">')
    html_parts.append('  <div class="navbar-content">')
    
    # Logo section
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
    
    # Navigation buttons
    html_parts.append('    <div class="nav-buttons-container">')
    
    # Home button
    home_active = "active" if current_page == 'home' else ""
    html_parts.append(f'      <a href="#" class="navbar-btn {home_active}" id="nav-home"> Home</a>')
    
    # Info button
    info_active = "active" if current_page == 'info' else ""
    html_parts.append(f'      <a href="#" class="navbar-btn {info_active}" id="nav-info"> Info</a>')
    
    html_parts.append('    </div>')
    html_parts.append('  </div>')
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)

def inject_navbar_js():
    """Inject JavaScript untuk navbar secara terpisah."""
    st.markdown("""
    <script>
    // Setup navbar buttons
    function setupNavbar() {
        // Home button
        const homeBtn = document.getElementById('nav-home');
        if (homeBtn) {
            homeBtn.onclick = function(e) {
                e.preventDefault();
                // Simpan ke session storage untuk Streamlit
                sessionStorage.setItem('nav_to', 'home');
                // Trigger page reload
                window.location.reload();
            };
        }
        
        // Info button
        const infoBtn = document.getElementById('nav-info');
        if (infoBtn) {
            infoBtn.onclick = function(e) {
                e.preventDefault();
                // Simpan ke session storage untuk Streamlit
                sessionStorage.setItem('nav_to', 'info');
                // Trigger page reload
                window.location.reload();
            };
        }
    }
    
    // Jalankan setup saat DOM siap
    document.addEventListener('DOMContentLoaded', setupNavbar);
    
    // Juga jalankan setelah timeout untuk memastikan
    setTimeout(setupNavbar, 100);
    </script>
    """, unsafe_allow_html=True)

# --- Page Render Functions ---

def render_navbar(current_page='home'):
    """Render fixed navbar dengan logo dan tombol sepenuhnya dalam HTML."""
    
    # Buat HTML navbar
    navbar_html = create_navbar_html(current_page)
    
    # Render navbar HTML
    st.markdown(navbar_html, unsafe_allow_html=True)
    
    # Inject JavaScript secara terpisah
    inject_navbar_js()
    
    # Buka main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Cek session storage untuk navigasi
    try:
        # Gunakan JavaScript untuk membaca session storage
        if 'nav_to' in st.session_state:
            nav_to = st.session_state.nav_to
            if nav_to and nav_to in ['home', 'info']:
                next_page(nav_to)
                # Clear setelah digunakan
                del st.session_state.nav_to
    except:
        pass

def close_navbar():
    """Close the navbar HTML structure."""
    st.markdown("</div>", unsafe_allow_html=True)

def render_candidate_form():
    """Render form untuk input data kandidat."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown("""
    <div class="candidate-form-container">
        <h1 class="candidate-form-title">Candidate Information</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Form untuk input data kandidat
    with st.container():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown("Please fill in your details before starting the interview.")
            
            # Form input menggunakan Streamlit
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
                    
                    # Generate interview ID
                    interview_id = str(uuid.uuid4())[:8].upper()
                    
                    # FIX: Get local time for Indonesia (WIB/WITA/WIT)
                    try:
                        # Method 1: Try to get system time (works if server is in Indonesia)
                        now_local = get_local_time_indonesia()
                        
                        # Method 2: If system time is UTC, adjust manually
                        # Add 7 hours for WIB (UTC+7), 8 for WITA (UTC+8), 9 for WIT (UTC+9)
                        # Uncomment line below if you need to adjust manually:
                        # from datetime import timedelta
                        # now_local = datetime.now() + timedelta(hours=7)  # WIB (UTC+7)
                        
                    except Exception as e:
                        now_local = get_local_time_indonesia()
                    
                    # Format the date
                    start_time = now_local.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Simpan data kandidat ke session state
                    st.session_state.candidate_data = {
                        'id': interview_id,
                        'name': name.strip(),
                        'email': email.strip(),
                        'start_time': start_time  # Use the corrected local time
                    }
                    st.session_state.interview_id = interview_id
                    
                    # Reset data interview
                    st.session_state.answers = {}
                    st.session_state.results = None
                    st.session_state.current_q = 1
                    
                    # Redirect ke halaman interview
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
            
            st.markdown("""
            <div class="info-card">
                <strong>Privacy Assurance:</strong>
                <p style="margin-top: 10px;">
                    Your data is securely processed and will not be shared with third parties.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tombol back
    if st.button("Back to Home", use_container_width=True):
        next_page('home')
    
    close_navbar()

def render_home_page():
    """Render the fixed landing page."""
    inject_global_css()
    
    # Render navbar dengan semua elemen dalam HTML
    render_navbar('home')
    
    # HERO SECTION
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job with comprehensive evaluation and actionable insights.</p>', unsafe_allow_html=True)
    
    # Ganti tombol "Start Interview Now" untuk redirect ke form kandidat
    if st.button("Start Interview Now", key="hero_start", type="primary"):
        next_page("candidate_form")
    
    st.markdown('</section>', unsafe_allow_html=True)
    
    # HOW IT WORKS SECTION
    st.markdown('<div class="text-center" style="margin-bottom: 40px;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How To Use</h2>', unsafe_allow_html=True)
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

    # Menggunakan loop biasa untuk flexbox yang lebih baik
    for num, title, desc in steps:
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
    st.markdown('<h2 class="section-title">Key Features</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="features-grid">', unsafe_allow_html=True)
    
    features = [
        ("🎤", "Advanced Speech-to-Text", "High-accuracy audio transcription using Whisper AI model"),
        ("📊", "Comprehensive Analysis", "Evaluate content, structure, and non-verbal aspects simultaneously"),
        ("⚡", "Real-time Feedback", "Instant evaluation results and personalized recommendations"),
        ("🎯", "Rubric-based Scoring", "Objective assessment against industry-standard interview rubrics"),
        ("📈", "Progress Tracking", "Monitor improvement across multiple practice sessions"),
        ("🔒", "Privacy Focused", "Your data is processed securely and not stored permanently")
    ]
    
    # Create two rows of features
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
    
    # Tombol Back to Home menggunakan Streamlit
    if st.button("🏠 Back to Home", type="primary"):
        next_page('home')
    
    close_navbar()

def render_interview_page():
    """Render the interview page dengan banner kandidat."""
    inject_global_css()
    render_navbar('interview')
    
    # Tampilkan informasi kandidat jika ada
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file_data = st.session_state.answers.get(q_id_str)
    
    # --- File Upload Logic ---
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
    
    # --- Navigation Controls ---
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
    
    # Tampilkan informasi kandidat jika ada
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
                            "confidence_score": f"{final_confidence_score*100:.2f}",
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

def render_final_summary_page():
    """Render the final results page dengan data kandidat."""
    inject_global_css()
    render_navbar('final_summary')
    
    # Header dengan informasi kandidat
    if st.session_state.candidate_data:
        candidate = st.session_state.candidate_data
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 25px; border-radius: 15px; margin-bottom: 30px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 28px;">Interview Report</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Final Evaluation Summary</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: bold;">{candidate['name']}</div>
                    <div style="font-size: 14px; opacity: 0.9;">ID: {candidate['id']}</div>
                    <div style="font-size: 14px; opacity: 0.9;">Date: {candidate['start_time']}</div>
                </div>
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
        all_confidence = [float(res['confidence_score'].split(' ')[0].replace('%', '')) 
                         for res in st.session_state.results.values()]
        
        all_tempo = []
        all_pause = []
        for res in st.session_state.results.values():
            tempo_str = res['non_verbal'].get('tempo_bpm', '0').split(' ')[0]
            pause_str = res['non_verbal'].get('total_pause_seconds', '0').split(' ')[0]
            try:
                all_tempo.append(float(tempo_str))
            except ValueError:
                all_tempo.append(0)
            try:
                all_pause.append(float(pause_str))
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
    
    # Tampilkan metric cards
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
            <div class="metric-value tempo-color">{avg_tempo:.1f}</div>
            <div class="metric-label">Average Tempo (BPM)</div>
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
    with st.expander("📋 View Detailed Breakdown by Question"):
        for q_key, res in st.session_state.results.items():
            q_num = q_key.replace('q', '')
            
            st.markdown(f"### Question {q_num}")
            st.write(f"**Question:** {res['question']}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Rubric Score", f"{res['final_score']}/4")
            with col_b:
                st.metric("Confidence", f"{res['confidence_score']}%")
            with col_c:
                st.metric("Non-Verbal Analysis", res['non_verbal'].get('qualitative_summary', 'N/A'))
            
            st.markdown("**Evaluation:**")
            st.info(res['rubric_reason'])
            
            with st.expander("View Transcript"):
                st.code(res['transcript'], language='text')
            
            st.markdown("---")
    
    # Action buttons
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔄 New Interview", use_container_width=True, type="primary"):
            # Reset data interview tapi pertahankan data kandidat jika mau
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page('candidate_form')
    
    with col_btn2:
        if st.button("📥 Download Report", use_container_width=True):
            # Tambahkan data kandidat ke report
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
                # Konversi ke JSON untuk download
                json_report = json.dumps(report_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Report as JSON",
                    data=json_report,
                    file_name=f"interview_report_{st.session_state.candidate_data['id']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("Download feature requires candidate information.")
    
    with col_btn3:
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