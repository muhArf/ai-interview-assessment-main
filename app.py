# app.py
import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# Tambahkan direktori saat ini dan 'models' ke PATH
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import logic dari folder models
try:
    # PENTING: Pastikan file di folder models/ sudah terimplementasi dengan benar.
    # Dummy imports jika modul di folder models/ belum terimplementasi sepenuhnya
    def load_stt_model(): return "STT_Model_Loaded"
    def load_text_models(): return None, None, None
    def load_embedder_model(): return "Embedder_Model_Loaded"
    def video_to_wav(video_path, audio_path): pass
    def noise_reduction(audio_path_in, audio_path_out): pass
    def transcribe_and_clean(audio_path, stt_model, spell_checker, embedder_model, english_words): return "This is a dummy transcript for testing.", 0.95
    def compute_confidence_score(transcript, log_prob_raw): return 0.95
    def analyze_non_verbal(audio_path): return {'tempo_bpm': '135 BPM', 'total_pause_seconds': '5.2', 'qualitative_summary': 'Normal pace'}
    def score_with_rubric(q_key_rubric, q_text, transcript, RUBRIC_DATA, embedder_model): return 4, "Excellent relevance and structural clarity."
    
    # Ganti dengan import yang sebenarnya jika modul sudah ada
    # from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
    # from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
    # from models.nonverbal_analysis import analyze_non_verbal
except ImportError as e:
    st.error(f"Failed to load modules from the 'models' folder. Ensure the folder structure and files are correct. Error: {e}")
    # st.stop() 

# Konfigurasi Halaman & Load Data
st.set_page_config(
    page_title="SEI-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inisialisasi State Awal
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_q' not in st.session_state:
    st.session_state.current_q = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'results' not in st.session_state:
    st.session_state.results = None

# Konstanta
TOTAL_QUESTIONS = 5
VIDEO_MAX_SIZE_MB = 50

# --- Fungsi Utility ---
def next_page(page_name):
    st.session_state.page = page_name
    st.rerun()

@st.cache_resource
def get_models():
    """Load semua model berat (hanya sekali)."""
    try:
        # PENTING: Ganti path model sesuai dengan implementasi Anda yang sebenarnya
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

# Load model di awal
STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

@st.cache_data
def load_questions():
    """Memuat pertanyaan dari questions.json."""
    try:
        # Dummy data jika questions.json tidak ada
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
    """Memuat data rubrik dari rubric_data.json."""
    try:
        # Dummy data jika rubric_data.json tidak ada
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

# --- Page Render Functions ---

def inject_custom_css():
    """Menyuntikkan CSS kustom untuk meniru desain Landing Page & Laporan."""
    st.markdown("""
    <style>
    /* 1. Reset Global dan Kontrol Padding (DIPERKUAT) */
    .stApp {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
    }
    /* Sembunyikan Header dan Footer Streamlit Default */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 2. Styling untuk Elemen Kustom - SIMPLIFIED */
    .custom-header {
        background-color: white;
        padding: 0 20px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Header yang lebih sederhana */
    .header-nav {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 15px;
    }

    /* HERO SECTION - Disederhanakan */
    .hero-section {
        padding: 60px 20px;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1a1a1a;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 30px;
        line-height: 1.6;
    }

    /* Steps section yang lebih sederhana */
    .step-card {
        background: white;
        border-radius: 8px;
        padding: 25px 15px;
        text-align: center;
        border: 1px solid #eee;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 220px;
        position: relative;
    }
    .step-number {
        width: 40px;
        height: 40px;
        background: #1a1a1a;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px;
        font-weight: bold;
    }
    .step-title {
        font-size: 16px;
        font-weight: 600;
        margin: 10px 0;
        color: #333;
    }
    .step-description {
        font-size: 13px;
        color: #666;
        line-height: 1.5;
    }

    /* Footer yang lebih sederhana */
    .custom-footer {
        background: #f8f9fa;
        padding: 20px;
        text-align: center;
        margin-top: 40px;
        border-top: 1px solid #eee;
    }

    /* Tombol Utama */
    .stButton>button {
        border-radius: 30px !important;
        padding: 12px 30px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        background-color: #1a1a1a !important;
        color: white !important;
        border: none !important;
    }
    
    /* === CARD METRIK HORIZONTAL === */
    .metric-grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    }
    
    .modern-metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    
    .card-value {
        font-size: 24px;
        font-weight: 700;
        line-height: 1.2;
    }
    .card-label {
        font-size: 11px;
        color: #666;
        margin-top: 5px;
    }
    
    </style>
    """, unsafe_allow_html=True)


def render_home_page():
    # 1. Suntikkan CSS kustom
    inject_custom_css()

    # --- 1. Header (Navbar Kustom) - DIHAPUS TOMBOL START INTERVIEW ---
    with st.container():
        st.markdown('<div class="custom-header">', unsafe_allow_html=True)
        
        col_logo, col_nav = st.columns([1, 4])
        
        with col_logo:
            try:
                st.image('assets/seiai.png', width=60) 
            except:
                st.markdown('<div style="font-weight: bold; font-size: 22px; color: #1a1a1a;">SEI-AI</div>', unsafe_allow_html=True)

        with col_nav:
            # HANYA TOMBOL "ABOUT" SAJA, TIDAK ADA "START INTERVIEW"
            if st.button("About", key="nav_info"):
                next_page('info')
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. Hero Section ---
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job.</p>', unsafe_allow_html=True)
    
    # Tombol Start Interview HANYA di sini
    if st.button("▶️ Start Interview", key="hero_start", type="primary"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page('interview')
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. How To Use Section ---
    st.markdown("---")
    st.markdown('<h2 style="text-align: center; margin: 40px 0;">How To Use</h2>', unsafe_allow_html=True)
    
    # Langkah-langkah dalam 5 kolom
    cols = st.columns(5)
    steps = [
        ("1", "Upload Answer Video", "Upload your video answer for each question."),
        ("2", "AI Processes Data", "AI processes video into transcript and analyzes non-verbal aspects."),
        ("3", "Semantic Scoring", "Answer is compared to ideal rubric for semantic scoring."),
        ("4", "Get Instant Feedback", "Receive score, rationale, and communication analysis."),
        ("5", "Improve Skills", "Use recommendations to practice and improve.")
    ]
    
    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f'''
            <div class="step-card">
                <div class="step-number">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-description">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)

    # --- 4. Footer ---
    st.markdown('<div class="custom-footer">', unsafe_allow_html=True)
    st.markdown("**SEI-AI Interviewer** • AI-powered interview preparation platform")
    st.markdown('<div style="font-size: 12px; color: #888; margin-top: 10px;">© 2024 SEI-AI. All Rights Reserved.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_info_page():
    st.title("📋 SEI-AI Application Information")
    st.markdown("""
    This application uses Machine Learning and Natural Language Processing (NLP) technologies 
    to analyze video interview answers.
    
    ### Analysis Process
    1. **Speech-to-Text (STT):** Your answer transcript is generated using advanced models.
    2. **Text Cleaning:** The transcribed text is corrected for spelling errors.
    3. **Non-Verbal Analysis:** Audio is analyzed for Speaking Tempo (BPM) and pause time.
    4. **Semantic Scoring:** Answers are scored based on comparison with ideal rubric points.
    
    ### Video Requirements
    * **Duration:** 30-60 seconds per answer recommended
    * **Format:** MP4, MOV, or WebM
    * **Maximum Size:** 50MB
    """)
    
    if st.button("🏠 Back to Home"):
        next_page('home')


def render_interview_page():
    st.title(f"Interview Question {st.session_state.current_q} of {TOTAL_QUESTIONS}")
    
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

    st.subheader("Question:")
    st.info(question_text)
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file_data = st.session_state.answers.get(q_id_str)
    
    with col_upload:
        uploaded_file = None
        
        if current_uploaded_file_data is None:
            uploaded_file = st.file_uploader(
                f"Upload Video Answer for Question {q_num} (Max {VIDEO_MAX_SIZE_MB}MB)",
                type=['mp4', 'mov', 'webm'],
                key=f"uploader_{q_id_str}"
            )
            
            if uploaded_file:
                if uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
                    st.error(f"File size exceeds the {VIDEO_MAX_SIZE_MB}MB limit.")
                    uploaded_file = None
                else:
                    st.session_state.answers[q_id_str] = uploaded_file
                    current_uploaded_file_data = uploaded_file 
                    st.success("File successfully uploaded.")
                    st.rerun()
        
        if current_uploaded_file_data:
            st.video(current_uploaded_file_data, format=current_uploaded_file_data.type)
            st.info(f"Answer video for Q{q_num} loaded: **{current_uploaded_file_data.name}**")
            
            if st.button("Delete/Re-upload Video", key=f"delete_q{q_num}", type="secondary"):
                if q_id_str in st.session_state.answers:
                    del st.session_state.answers[q_id_str]
                if f"uploader_{q_id_str}" in st.session_state:
                     del st.session_state[f"uploader_{q_id_str}"]
                st.rerun()
        else:
            st.warning("Please upload your answer file to continue.")

    with col_control:
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("Next ⏩", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("Finish ▶️", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')

        if q_num > 1:
            if st.button("⏪ Previous", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

        if st.button("🏠 Home", use_container_width=True):
            next_page('home')


def render_processing_page():
    st.title("⚙️ Processing Your Answers")
    st.info("Please wait while we analyze your video responses. This may take a few minutes.")

    if st.session_state.results is not None and st.session_state.results != {}:
        next_page('final_summary') 
        return

    if st.session_state.results is None:
        results = {}
        progress_bar = st.progress(0, text="Starting analysis...")
        
        if not all([STT_MODEL, EMBEDDER_MODEL]):
            st.error("Core models failed to load.")
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
                        progress_bar.progress((i-1)*20, text=f"Processing Q{i}...")
                        
                        temp_video_path = os.path.join(temp_dir, f'video_{q_key_rubric}.mp4')
                        temp_audio_path = os.path.join(temp_dir, f'audio_{q_key_rubric}.wav') 
                        
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_file.getbuffer())

                        video_to_wav(temp_video_path, temp_audio_path)
                        noise_reduction(temp_audio_path, temp_audio_path)
                        
                        transcript, log_prob_raw = transcribe_and_clean(
                            temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        final_confidence_score_0_1 = compute_confidence_score(transcript, log_prob_raw)
                        non_verbal_res = analyze_non_verbal(temp_audio_path)
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[Score calculation error] {reason}"
                        
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value,
                            "rubric_reason": reason,
                            "confidence_score": f"{final_confidence_score_0_1*100:.2f}",
                            "non_verbal": non_verbal_res
                        }

                        progress_bar.progress(i*20, text=f"Q{i} complete")
                    else:
                        st.warning(f"Skipping Q{i}: Missing data")
                
                st.session_state.results = results
                progress_bar.progress(100, text="Analysis complete!")
                st.success("All answers processed successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"Error during processing: {e}")
            progress_bar.empty()
            st.session_state.results = None 
            if st.button("🏠 Back to Home"):
                 st.session_state.clear() 
                 next_page('home')
            return


def render_results_page():
    if not st.session_state.results:
        st.error("Results not found.")
        if st.button("🏠 Back to Home"):
            st.session_state.clear()
            next_page('home')
        return
        
    next_page('final_summary')


def render_final_summary_page():
    inject_custom_css() 
    
    st.title("🏆 Final Evaluation Report")
    st.markdown("---")

    if not st.session_state.results:
        st.error("Result data not found.")
        if st.button("Back to Home"):
            next_page('home') 
        return

    # Metrics Calculation
    try:
        all_scores = [int(res['final_score']) for res in st.session_state.results.values()]
        all_confidence = [float(res['confidence_score'].split(' ')[0].replace('%', '')) for res in st.session_state.results.values()]
        
        all_tempo = []
        all_pause = []
        for res in st.session_state.results.values():
             tempo_str = res['non_verbal'].get('tempo_bpm', '0').split(' ')[0]
             pause_str = res['non_verbal'].get('total_pause_seconds', '0').split(' ')[0]
             try:
                 all_tempo.append(float(tempo_str))
             except:
                 all_tempo.append(0)
             try:
                 all_pause.append(float(pause_str))
             except:
                 all_pause.append(0)

        avg_score = np.mean(all_scores) if all_scores else 0
        avg_confidence = np.mean(all_confidence) if all_confidence else 0
        avg_tempo = np.mean(all_tempo) if all_tempo else 0
        total_pause = np.sum(all_pause) 
    
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return

    # Performance Summary
    st.subheader("📊 Performance Summary")
    
    st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div style="font-size: 24px; font-weight: bold; color: #2ecc71;">{avg_score:.1f}/4</div>
        <div class="card-label">Avg. Content Score</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div style="font-size: 24px; font-weight: bold; color: #3498db;">{avg_confidence:.1f}%</div>
        <div class="card-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div style="font-size: 24px; font-weight: bold; color: #f39c12;">{avg_tempo:.0f}</div>
        <div class="card-label">Avg. Tempo (BPM)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="modern-metric-card">
        <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">{total_pause:.1f}</div>
        <div class="card-label">Total Pause (sec)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Recommendations
    st.subheader("💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Content Performance:**")
        if avg_score >= 3.5:
            st.success("Excellent content quality and relevance.")
        elif avg_score >= 2.5:
            st.info("Good content, room for improvement in depth.")
        else:
            st.warning("Focus on aligning answers with rubric criteria.")
    
    with col2:
        st.markdown("**Delivery Performance:**")
        if 125 <= avg_tempo <= 150:
            st.success("Good speaking pace.")
        else:
            st.warning(f"Consider adjusting pace (optimal: 125-150 BPM). Current: {avg_tempo:.0f} BPM")
        
        if total_pause > 120:
            st.warning("High pause time detected. Practice smoother delivery.")

    # Detailed Results
    with st.expander("📋 View Detailed Results per Question"):
        for q_key, res in st.session_state.results.items():
            q_num = q_key.replace('q', '')
            st.markdown(f"**Question {q_num}:** {res['question'][:100]}...")
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Score", f"{res['final_score']}/4")
            with cols[1]:
                st.metric("Accuracy", res['confidence_score'])
            with cols[2]:
                st.metric("Tempo", res['non_verbal'].get('tempo_bpm', 'N/A'))
            with cols[3]:
                st.metric("Pause", res['non_verbal'].get('total_pause_seconds', 'N/A'))
            
            if st.checkbox(f"Show transcript Q{q_num}", key=f"transcript_{q_num}"):
                st.text_area("Transcript", res['transcript'], height=100)
            
            st.markdown("---")

    # Action Button
    if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
        st.session_state.clear() 
        next_page('home')
    
    if st.button("🏠 Back to Home", use_container_width=True):
        next_page('home')


# Main App Execution Flow
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'info':
    render_info_page()
elif st.session_state.page == 'interview':
    render_interview_page()
elif st.session_state.page == 'processing':
    render_processing_page()
elif st.session_state.page == 'results':
    render_results_page() 
elif st.session_state.page == 'final_summary':
    render_final_summary_page()