# app.py (versi dengan laporan invoice style + auto print)

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

# --- Page Render Functions ---

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

def render_candidate_form():
    """Render form untuk input data kandidat."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.08); margin: 40px auto; max-width: 800px; border: 1px solid #f0f0f0;">
        <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 30px; color: #000000; text-align: center;">Candidate Information</h1>
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
            <div style="background: #f8f9ff; border-left: 4px solid #667eea; padding: 15px; border-radius: 8px; margin: 20px 0;">
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

def render_home_page():
    """Render the fixed landing page."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job with comprehensive evaluation and actionable insights.</p>', unsafe_allow_html=True)
    
    if st.button("Start Interview Now", key="hero_start", type="primary"):
        next_page("candidate_form")
    
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
    """)
    
    if st.button("🏠 Back to Home", type="primary"):
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

# --- Fungsi untuk generate Print-Ready HTML Report ---
def generate_invoice_report(candidate_data, results, metrics):
    """Generate professional invoice-style report with auto-print."""
    from datetime import datetime
    
    # Format tanggal yang lebih baik
    interview_date = candidate_data.get('start_time', 'N/A')
    report_date = datetime.now().strftime("%B %d, %Y")
    
    # Hitung total score
    total_score = sum(int(res['final_score']) for res in results.values())
    max_score = len(results) * 4
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Interview Assessment Report - {candidate_data.get('name', 'Candidate')}</title>
        <style>
            @media print {{
                @page {{
                    size: A4;
                    margin: 20mm;
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    background: white !important;
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                }}
                .no-print {{
                    display: none !important;
                }}
                .page-break {{
                    page-break-before: always;
                }}
            }}
            
            @media screen {{
                body {{
                    background: #f5f5f5;
                    padding: 20px;
                }}
                .report-container {{
                    max-width: 210mm;
                    margin: 0 auto;
                    box-shadow: 0 0 30px rgba(0,0,0,0.1);
                }}
            }}
            
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            
            .report-container {{
                background: white;
                min-height: 297mm;
                position: relative;
            }}
            
            /* HEADER STYLE */
            .header {{
                padding: 40px 50px 30px 50px;
                border-bottom: 3px double #2c3e50;
                position: relative;
            }}
            
            .company-info {{
                float: left;
                width: 60%;
            }}
            
            .company-name {{
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                margin: 0 0 5px 0;
                letter-spacing: 1px;
            }}
            
            .company-tagline {{
                font-size: 14px;
                color: #7f8c8d;
                font-style: italic;
                margin: 0;
            }}
            
            .report-info {{
                float: right;
                text-align: right;
                width: 35%;
            }}
            
            .report-title {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 0 0 15px 0;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            
            .report-id {{
                font-size: 14px;
                color: #7f8c8d;
                margin: 5px 0;
            }}
            
            .report-date {{
                font-size: 14px;
                color: #7f8c8d;
                margin: 5px 0;
            }}
            
            /* CANDIDATE INFO STYLE */
            .candidate-section {{
                padding: 30px 50px;
                background: #f8f9fa;
                margin: 20px 50px;
                border-radius: 8px;
                border-left: 5px solid #2c3e50;
            }}
            
            .section-title {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
                border-bottom: 2px solid #eee;
                padding-bottom: 8px;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }}
            
            .info-item {{
                margin-bottom: 10px;
            }}
            
            .info-label {{
                font-weight: bold;
                color: #7f8c8d;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 3px;
            }}
            
            .info-value {{
                font-size: 16px;
                color: #2c3e50;
            }}
            
            /* SUMMARY METRICS */
            .summary-section {{
                padding: 0 50px 30px 50px;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-top: 20px;
            }}
            
            .metric-box {{
                text-align: center;
                padding: 25px 15px;
                border-radius: 8px;
                border: 2px solid #eee;
                background: white;
                position: relative;
                box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            }}
            
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
                font-family: 'Arial', sans-serif;
            }}
            
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }}
            
            .metric-note {{
                font-size: 11px;
                color: #95a5a6;
                font-style: italic;
                margin-top: 5px;
            }}
            
            /* QUESTION DETAILS */
            .questions-section {{
                padding: 30px 50px;
            }}
            
            .question-card {{
                border: 1px solid #eee;
                border-radius: 8px;
                margin-bottom: 25px;
                overflow: hidden;
                background: white;
                box-shadow: 0 3px 15px rgba(0,0,0,0.05);
            }}
            
            .question-header {{
                background: linear-gradient(to right, #f8f9fa, #e9ecef);
                padding: 20px;
                border-bottom: 2px solid #dee2e6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .question-title {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin: 0;
            }}
            
            .question-score {{
                background: #2c3e50;
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 16px;
            }}
            
            .question-body {{
                padding: 25px;
            }}
            
            .question-text {{
                font-size: 16px;
                color: #34495e;
                margin-bottom: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                font-style: italic;
            }}
            
            .metrics-row {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 25px;
            }}
            
            .metric-small {{
                text-align: center;
                padding: 15px 10px;
                background: #f8f9fa;
                border-radius: 6px;
                border: 1px solid #e9ecef;
            }}
            
            .metric-small-title {{
                font-weight: bold;
                color: #7f8c8d;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}
            
            .metric-small-value {{
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                font-family: 'Arial', sans-serif;
            }}
            
            .evaluation-box {{
                background: #e8f4fc;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }}
            
            .evaluation-title {{
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .transcript-box {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                border: 1px solid #e9ecef;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
                max-height: 150px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            
            /* FOOTER */
            .footer {{
                padding: 30px 50px;
                border-top: 3px double #2c3e50;
                margin-top: 40px;
                text-align: center;
                color: #7f8c8d;
                font-size: 12px;
            }}
            
            .footer-note {{
                font-style: italic;
                margin-top: 10px;
                font-size: 11px;
            }}
            
            .signature-section {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
            
            .signature-line {{
                width: 200px;
                border-top: 1px solid #7f8c8d;
                margin: 30px auto 10px auto;
            }}
            
            .signature-label {{
                text-align: center;
                font-size: 12px;
                color: #7f8c8d;
            }}
            
            /* PRINT BUTTON */
            .print-button {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #2c3e50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                z-index: 1000;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            }}
            
            .print-button:hover {{
                background: #34495e;
            }}
            
            .clearfix::after {{
                content: "";
                clear: both;
                display: table;
            }}
        </style>
        <script>
            function autoPrint() {{
                window.print();
                setTimeout(function() {{
                    window.close();
                }}, 1000);
            }}
            
            window.onload = function() {{
                setTimeout(autoPrint, 500);
            }};
        </script>
    </head>
    <body>
        <button class="print-button no-print" onclick="window.print()">🖨️ Print Report</button>
        
        <div class="report-container">
            <!-- HEADER -->
            <div class="header clearfix">
                <div class="company-info">
                    <h1 class="company-name">SEI-AI INTERVIEW SYSTEM</h1>
                    <p class="company-tagline">Artificial Intelligence Powered Interview Assessment</p>
                </div>
                <div class="report-info">
                    <h2 class="report-title">ASSESSMENT REPORT</h2>
                    <p class="report-id"><strong>Report ID:</strong> {candidate_data.get('id', 'N/A')}</p>
                    <p class="report-date"><strong>Date:</strong> {report_date}</p>
                </div>
            </div>
            
            <!-- CANDIDATE INFORMATION -->
            <div class="candidate-section">
                <div class="section-title">Candidate Information</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Full Name</div>
                        <div class="info-value">{candidate_data.get('name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Interview ID</div>
                        <div class="info-value">{candidate_data.get('id', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Email Address</div>
                        <div class="info-value">{candidate_data.get('email', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Interview Date</div>
                        <div class="info-value">{interview_date}</div>
                    </div>
                </div>
            </div>
            
            <!-- PERFORMANCE SUMMARY -->
            <div class="summary-section">
                <div class="section-title">Performance Summary</div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('avg_score', 0):.2f}<span style="font-size: 16px; color: #95a5a6;">/4.0</span></div>
                        <div class="metric-label">Average Score</div>
                        <div class="metric-note">Content Quality</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('avg_confidence', 0):.1f}<span style="font-size: 16px; color: #95a5a6;">%</span></div>
                        <div class="metric-label">Confidence Level</div>
                        <div class="metric-note">Transcript Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('avg_tempo', 0):.0f}</div>
                        <div class="metric-label">Speaking Tempo</div>
                        <div class="metric-note">Words per minute</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{metrics.get('total_pause', 0):.1f}<span style="font-size: 16px; color: #95a5a6;">s</span></div>
                        <div class="metric-label">Total Pauses</div>
                        <div class="metric-note">Speaking rhythm</div>
                    </div>
                </div>
            </div>
            
            <!-- QUESTION DETAILS -->
            <div class="questions-section">
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
                    <div class="question-body">
                        <div class="question-text">"{result.get('question', 'N/A')}"</div>
                        
                        <div class="metrics-row">
                            <div class="metric-small">
                                <div class="metric-small-title">Confidence</div>
                                <div class="metric-small-value">{result.get('confidence_score', '0%')}</div>
                            </div>
                            <div class="metric-small">
                                <div class="metric-small-title">Tempo</div>
                                <div class="metric-small-value">{tempo_value} BPM</div>
                            </div>
                            <div class="metric-small">
                                <div class="metric-small-title">Pause</div>
                                <div class="metric-small-value">{pause_value}s</div>
                            </div>
                            <div class="metric-small">
                                <div class="metric-small-title">Delivery</div>
                                <div class="metric-small-value">{non_verbal.get('qualitative_summary', 'N/A')[:15]}...</div>
                            </div>
                        </div>
                        
                        <div class="evaluation-box">
                            <div class="evaluation-title">Evaluation</div>
                            <div>{result.get('rubric_reason', 'No rationale provided.')}</div>
                        </div>
                        
                        <div style="margin-top: 20px;">
                            <div class="evaluation-title">Transcript</div>
                            <div class="transcript-box">{result.get('transcript', 'No transcript available.')}</div>
                        </div>
                    </div>
                </div>
        """
    
    html += f"""
            </div>
            
            <!-- FOOTER -->
            <div class="footer">
                <div style="margin-bottom: 20px;">
                    <p><strong>Overall Assessment:</strong> {total_score}/{max_score} total points achieved</p>
                </div>
                
                <div class="signature-section">
                    <div class="signature-line"></div>
                    <div class="signature-label">SEI-AI Interview System</div>
                </div>
                
                <div class="footer-note">
                    <p>This report was automatically generated by the SEI-AI Interview Assessment System.</p>
                    <p>Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
                    <p>Confidential Document - For authorized use only</p>
                </div>
            </div>
        </div>
        
        <script>
            // Auto-print after page loads
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
    
    # Action buttons - TOMBOL UTAMA UNTUK PRINT
    st.markdown("---")
    st.subheader("📄 Download Report")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔄 New Interview", use_container_width=True, type="primary"):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page('candidate_form')
    
    with col_btn2:
        if st.button("📥 JSON Data", use_container_width=True):
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
                    label="Download JSON",
                    data=json_report,
                    file_name=f"interview_report_{st.session_state.candidate_data['id']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("Download feature requires candidate information.")
    
    with col_btn3:
        # TOMBOL UTAMA UNTUK PRINT LAPORAN
        if st.button("🖨️ Print Report", use_container_width=True, type="primary"):
            if st.session_state.candidate_data:
                metrics = {
                    'avg_score': avg_score,
                    'avg_confidence': avg_confidence,
                    'avg_tempo': avg_tempo,
                    'total_pause': total_pause
                }
                
                html_bytes = generate_invoice_report(
                    st.session_state.candidate_data,
                    st.session_state.results,
                    metrics
                )
                
                # Buat popup window untuk print
                html_content = html_bytes.decode('utf-8')
                
                # Inject JavaScript untuk membuka popup dan auto-print
                js_code = f"""
                <script>
                function openPrintWindow() {{
                    var printWindow = window.open('', '_blank');
                    printWindow.document.write(`{html_content}`);
                    printWindow.document.close();
                    
                    // Auto print setelah window terbuka
                    printWindow.onload = function() {{
                        printWindow.print();
                        // Tutup window setelah print (opsional)
                        // setTimeout(function() {{ printWindow.close(); }}, 1000);
                    }};
                }}
                
                openPrintWindow();
                </script>
                """
                
                st.components.v1.html(js_code, height=0)
                
                st.success("Report opened in new window for printing. If popup blocked, please allow popups for this site.")
            else:
                st.info("Report generation requires candidate information.")
    
    # Tombol back to home
    st.markdown("---")
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