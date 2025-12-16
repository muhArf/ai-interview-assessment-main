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
    from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
    from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
    from models.nonverbal_analysis import analyze_non_verbal
except ImportError as e:
    st.error(f"Failed to load modules from the 'models' folder. Ensure the folder structure and files are correct. Error: {e}")
    st.stop() 

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

    /* 2. Styling untuk Elemen Kustom */
    .custom-header {
        background-color: white;
        padding: 0 50px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        height: 100px; /* Tinggi Navbar */
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    /* Perbaikan Navbar: Memastikan elemen Navigasi sejajar */
    .header-nav {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        width: 100%;
        gap: 10px; 
    }
    .header-nav button {
        margin-top: 0px !important;
        padding: 8px 15px !important;
        font-size: 14px !important;
        border-radius: 6px !important;
        height: 40px; 
    }
    /* Mengatasi Streamlit elements inside the column block */
    .header-nav > div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .header-nav p {
        margin: 0; 
        padding-top: 10px; /* Menyelaraskan teks 'Home' */
    }


    /* HERO SECTION */
    .hero-section {
        background-color: white;
        padding: 100px 50px;
        text-align: center;
    }
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .hero-subtitle {
        font-size: 18px;
        color: #5d5988;
        max-width: 600px;
        margin: 0 auto 40px auto;
    }

    /* Styling How To Use Steps */
    .steps-container {
        display: flex;
        flex-wrap: wrap;
        gap: 30px;
        justify-content: center;
        padding: 50px 0;
    }
    .step-card {
        background-color: #f9f9ff; 
        border-radius: 6px;
        padding: 40px 20px 20px 20px;
        text-align: center;
        position: relative;
        flex-grow: 1;
        max-width: 300px;
        min-height: 250px; 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); 
        transition: all 0.3s;
    }
    .step-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .step-number {
        position: absolute;
        top: -30px; 
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 60px;
        background-color: black;
        border-radius: 50%;
        color: white;
        font-size: 20px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
    }
    .step-title {
        font-size: 18px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .step-description {
        font-size: 14px;
        color: #5d5988;
    }

    /* Styling Footer */
    .custom-footer {
        background-color: black;
        color: #9795b4; 
        padding: 20px 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
        margin-top: 50px;
    }

    /* Tombol Utama */
    .stButton>button {
        border-radius: 40px !important;
        padding: 15px 40px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    .primary-btn-container button {
        background-color: black !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s;
    }
    .primary-btn-container button:hover {
        background-color: #333333 !important;
        transform: translateY(-2px);
    }
    
    /* === CARD METRIK HORIZONTAL (PERBAIKAN FONT DAN SPACING) === */
    .metric-grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr) !important; 
        gap: 10px !important; 
        margin-bottom: 30px;
        width: 100%; 
    }
    
    .modern-metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 15px !important; 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        text-align: left;
        transition: transform 0.2s;
        height: auto; 
        min-width: 0; 
    }
    
    .card-value-line {
        display: flex;
        flex-direction: row; 
        align-items: center;
        gap: 5px !important; 
        margin-bottom: 3px !important;
    }
    
    .card-value {
        font-size: 26px !important; 
        font-weight: 700;
        line-height: 1.1; 
    }
    .card-label {
        font-size: 12px !important; 
        color: #7f8c8d;
        font-weight: 500;
        margin-top: 3px; 
    }
    /* Warna untuk Skor dan Tempo */
    .score-color { color: #2ecc71; } 
    .accuracy-color { color: #3498db; } 
    .tempo-color { color: #f39c12; } 
    .pause-color { color: #e74c3c; } 
    
    }
    
    </style>
    """, unsafe_allow_html=True)


def render_home_page():
    # 1. Suntikkan CSS kustom
    inject_custom_css()

    # --- 1. Header (Navbar Kustom) ---
    with st.container():
        st.markdown('<div class="custom-header">', unsafe_allow_html=True)
        
        # Menggunakan dua kolom utama: Logo dan Navigasi
        col_logo, col_nav = st.columns([1, 4])
        
        with col_logo:
            # Logo/Nama Aplikasi
            try:
                st.image('assets/seiai.png', width=80, output_format='PNG') 
            except FileNotFoundError:
                st.markdown('<p style="font-weight: bold; font-size: 20px; margin-top: 10px;">SEI-AI</p>', unsafe_allow_html=True) 

        with col_nav:
            # Kontainer Navigasi dengan class 'header-nav' untuk styling khusus
            st.markdown('<div class="header-nav">', unsafe_allow_html=True)
            
            # Menggunakan 3 kolom di dalam col_nav: Home, Info, Start
            col_home, col_info, col_start = st.columns([0.5, 1, 1])
            
            with col_home:
                # Teks Home yang sejajar dengan tombol
                st.markdown('<p style="font-size: 14px; font-weight: 500; margin-top: 10px;">Home</p>', unsafe_allow_html=True)
            
            with col_info:
                # Tombol Info Aplikasi
                if st.button("App Info", key="nav_info", type="secondary"):
                    next_page('info')
            
            with col_start:
                # Tombol Mulai Wawancara di Navbar
                if st.button("Start Interview", key="nav_start", type="primary"):
                    st.session_state.answers = {}
                    st.session_state.results = None
                    st.session_state.current_q = 1
                    next_page('interview')
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # --- 2. Hero Section ---
    st.markdown('<section class="hero-section">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job.</p>', unsafe_allow_html=True)
    
    # Tombol Aksi Hero Section
    st.markdown('<div class="primary-btn-container" style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    if st.button("▶️ Start Interview", key="hero_start"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page('interview')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</section>', unsafe_allow_html=True)


    # --- 3. How To Use Section (Langkah-Langkah) ---
    st.markdown('<section style="padding: 50px; background-color: white;">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 40px; text-align: center; margin-bottom: 70px;">How To Use</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    
    # Grid Langkah-Langkah: Menggunakan kolom Streamlit untuk 5 card
    cols = st.columns(5)
    
    steps_data = [
        ("1", "Upload Answer Video", "Upload your video answer for each question given by the AI."),
        ("2", "AI Processes Data", "The AI will process the video into a transcript and analyze non-verbal aspects."),
        ("3", "Semantic Scoring", "Your answer is compared to an ideal rubric for semantic scoring (Content Relevance)."),
        ("4", "Get Instant Feedback", "Receive final score, rationale, and instant communication analysis."),
        ("5", "Improve Skills", "Use the recommendations to practice and improve until you are confident.")
    ]
    
    for i, (num, title, desc) in enumerate(steps_data):
        with cols[i]:
            # Struktur Card Kustom
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{num}</div>
                <h3 class="step-title">{title}</h3>
                <p class="step-description">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)


    # --- 4. Footer Kustom ---
    st.markdown('<div class="custom-footer">', unsafe_allow_html=True)
    col_footer_left, col_footer_right = st.columns(2)
    
    with col_footer_left:
        st.markdown('<p style="font-weight: bold; color: white; font-size: 16px;">SEI-AI Interviewer</p>', unsafe_allow_html=True)
    
    with col_footer_right:
        st.markdown('<p style="text-align: right;">Copyright © 2024 SEI-AI Interviewer. All Rights Reserved.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_info_page():
    st.title("SEI-AI Application Information")
    st.markdown("""
    This application uses Machine Learning and Natural Language Processing (NLP) technologies to analyze video interview answers.
    
    ### Analysis Process
    1. **Speech-to-Text (STT):** Your answer transcript is generated using the *Whisper* model.
    2. **Text Cleaning:** The transcribed text is corrected for spelling errors (spell check) and ambiguities.
    3. **Non-Verbal Analysis:** Audio is analyzed for Speaking Tempo (BPM) and total Pause time.
    4. **Semantic Scoring:** Answers are scored based on semantic comparison with ideal rubric points using the *Sentence-Transformer* model.
    
    ### Video Upload Requirements
    * **Duration:** Recommended 30-60 seconds per answer.
    * **Format:** MP4, MOV, or WebM.
    * **Maximum Size:** 50MB.
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

    st.header("Question:")
    st.info(question_text)
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file_data = st.session_state.answers.get(q_id_str)
    
    # --- Logic untuk Mengelola File Upload ---
    
    with col_upload:
        
        uploaded_file = None
        
        # Cek apakah file sudah ada di session_state. Jika tidak, tampilkan uploader.
        if current_uploaded_file_data is None:
            # Jika belum ada, tampilkan uploader
            uploaded_file = st.file_uploader(
                f"Upload Video Answer for Question {q_num} (Max {VIDEO_MAX_SIZE_MB}MB)",
                type=['mp4', 'mov', 'webm'],
                key=f"uploader_{q_id_str}"
            )
            
            # 1. Logic Save File Baru
            if uploaded_file:
                if uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
                    st.error(f"File size exceeds the {VIDEO_MAX_SIZE_MB}MB limit. The file will not be processed.")
                    uploaded_file = None
                else:
                    # Simpan objek file yang diunggah ke session state.
                    st.session_state.answers[q_id_str] = uploaded_file
                    current_uploaded_file_data = uploaded_file 
                    st.success("File successfully uploaded.")
                    # Rerun agar uploader menghilang dan hanya video yang muncul
                    st.rerun() 
        
        # 2. Logic Display/View File Lama (atau yang baru saja di-upload)
        if current_uploaded_file_data:
            st.video(current_uploaded_file_data, format=current_uploaded_file_data.type)
            st.info(f"Answer video for Q{q_num} loaded: **{current_uploaded_file_data.name}**")
            
            # Tombol untuk menghapus/mengunggah ulang file yang sudah ada
            if st.button("Delete/Re-upload Video", key=f"delete_q{q_num}", type="secondary"):
                # Hapus dari session state
                if q_id_str in st.session_state.answers:
                    del st.session_state.answers[q_id_str]
                # Hapus item dari key Streamlit uploader (Jika ada)
                if f"uploader_{q_id_str}" in st.session_state:
                     del st.session_state[f"uploader_{q_id_str}"]
                st.rerun()
                
        else:
            # Jika tidak ada file dan uploader tidak menghasilkan apa-apa
            st.warning("Please upload your answer file to continue.")

    # --- Logic Kontrol (Next/Previous) ---

    with col_control:
        
        
        # Kondisi 'ready' sekarang hanya bergantung pada session_state.answers
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("Next Question ⏩", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("Finish & Process ▶️", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')

        if q_num > 1:
            if st.button("⏪ Previous Question", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()


def render_processing_page():
    st.title("⚙️ Answer Analysis Process")
    st.info("Please wait, this process may take a few minutes depending on the video duration.")

    if st.session_state.results is not None and st.session_state.results != {}:
        # Redirect to the final summary page after processing
        next_page('final_summary') 
        return

    if st.session_state.results is None:
        
        results = {}
        progress_bar = st.progress(0, text="Starting Process...")
        
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
                        
                        # --- 1. Save Video 
                        progress_bar.progress((i-1)*10 + 1, text=f"Q{i}: Saving video...")
                        temp_video_path = os.path.join(temp_dir, f'video_{q_key_rubric}.mp4')
                        # Penambahan Path Audio Temporer
                        temp_audio_path = os.path.join(temp_dir, f'audio_{q_key_rubric}.wav') 
                        
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_file.getbuffer())

                        # --- 2. Audio Extraction & Noise Reduction
                        progress_bar.progress((i-1)*10 + 3, text=f"Q{i}: Extracting audio and Noise Reduction...")
                        
                        # Pastikan video_to_wav dipanggil dengan 2 argumen:
                        video_to_wav(temp_video_path, temp_audio_path)
                        
                        noise_reduction(temp_audio_path, temp_audio_path) 
                        
                        # --- 3. Speech-to-Text (STT) & Cleaning
                        progress_bar.progress((i-1)*10 + 5, text=f"Q{i}: Transcription and Text Cleaning...")
                        transcript, log_prob_raw = transcribe_and_clean(
                            temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        final_confidence_score_0_1 = compute_confidence_score(transcript, log_prob_raw)
                        
                        # --- 4. Non-Verbal Analysis
                        progress_bar.progress((i-1)*10 + 7, text=f"Q{i}: Non-Verbal Analysis...")
                        non_verbal_res = analyze_non_verbal(temp_audio_path)

                        # --- 5. Answer Scoring (Semantic)
                        progress_bar.progress((i-1)*10 + 9, text=f"Q{i}: Semantic Scoring...")
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[ERROR: Score failed to calculate/Wrong data type. Default score 0 used.] {reason}"
                        
                        # --- 6. Save Results
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value,
                            "rubric_reason": reason,
                            "confidence_score": f"{final_confidence_score_0_1*100:.2f}",
                            "non_verbal": non_verbal_res
                        }

                        progress_bar.progress(i*10, text=f"Q{i} Complete.")
                    else:
                        st.warning(f"Skipping Q{i} (ID: {q_id_str}): Answer file not uploaded or rubric data missing.")
                    
                st.session_state.results = results
                progress_bar.progress(100, text="Process Complete! Redirecting to Final Report.")
                # Pindah ke laporan akumulasi
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

def render_results_page():
    # Fungsi ini sekarang hanya pengalih/redirector
    
    if not st.session_state.results:
        st.error("Results not found. Please try processing again.")
        if st.button("🏠 Back to Home"):
            st.session_state.clear()
            next_page('home')
        return
        
    # REDIRECT DIRECTLY TO THE FINAL ACCUMULATED REPORT
    next_page('final_summary')


def render_final_summary_page():
    # Suntikkan CSS lagi untuk memastikan styling card berfungsi
    inject_custom_css() 
    
    st.title("🏆 Final Evaluation Report")
    st.markdown("---") # Pemisah tipis untuk judul

    if not st.session_state.results:
        st.error("Result data not found.")
        if st.button("Back"):
            next_page('home') 
        return

    # --- 1. Combined Metrics Calculation ---
    try:
        all_scores = [int(res['final_score']) for res in st.session_state.results.values()]
        # Extract and clean data, handling potential non-numeric strings
        all_confidence = [float(res['confidence_score'].split(' ')[0].replace('%', '')) for res in st.session_state.results.values()]
        
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

        # Calculate averages
        avg_score = np.mean(all_scores) if all_scores else 0
        avg_confidence = np.mean(all_confidence) if all_confidence else 0
        avg_tempo = np.mean(all_tempo) if all_tempo else 0
        total_pause = np.sum(all_pause) 
    
    except Exception as e:
        st.error(f"Failed to calculate accumulated metrics. Error: {e}")
        return

    # Qualitative logic (in English)
    def get_overall_comment(avg_score, avg_tempo):
        comment = []
        if avg_score >= 3.5:
            comment.append("The content and relevance of the answers were strong and well-structured.")
        elif avg_score >= 2.5:
            comment.append("Answer content was adequate, but could be improved for deeper material understanding.")
        else:
            comment.append("Answer content was less relevant to the questions; focus is needed on rubric alignment.")
        
        if avg_tempo > 150:
            comment.append("The overall speaking tempo tends to be too fast; practice slowing down for clarity.")
        elif avg_tempo < 125:
            comment.append("The overall speaking tempo is too slow, potentially losing interviewer attention.")
        else:
            comment.append("The overall speaking tempo is within the optimal range (125-150 BPM).")
        return " ".join(comment)

    # --- 2. Average Metrics Display (Minimalist Grid Cards - Horizontal Look) ---
    st.subheader("📊 Performance Summary")
    
    # Kunci: Gunakan div kustom dengan display: grid yang kuat.
    st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)
    
    # Card 1: Average Content Score (LABEL DIPENDEKKAN)
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-content-wrapper">
            <div class="card-value-line">
                <span class="card-icon score-color">🎯</span>
                <span class="card-value score-color">{avg_score:.2f} / 4</span>
            </div>
            <span class="card-label">Avg. Content Score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 2: Average Transcript Accuracy (LABEL DIPENDEKKAN)
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-content-wrapper">
            <div class="card-value-line">
                <span class="card-icon accuracy-color">🤖</span>
                <span class="card-value accuracy-color">{avg_confidence:.2f}%</span>
            </div>
            <span class="card-label">Avg. Accuracy (%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 3: Average Tempo (LABEL DIPENDEKKAN)
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-content-wrapper">
            <div class="card-value-line">
                <span class="card-icon tempo-color">⏱️</span>
                <span class="card-value tempo-color">{avg_tempo:.2f}</span>
            </div>
            <span class="card-label">Avg. Tempo (BPM)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Total Pause Time (LABEL DIPENDEKKAN)
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-content-wrapper">
            <div class="card-value-line">
                <span class="card-icon pause-color">⏸️</span>
                <span class="card-value pause-color">{total_pause:.2f}</span>
            </div>
            <span class="card-label">Total Pause (sec)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---") 

    # --- 3. Objective Evaluation and Recommendations ---
    st.subheader("💡 Objective Evaluation & Action Plan")
    
    col_eval, col_recom = st.columns(2)
    
    # Evaluation Box
    with col_eval:
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("### Performance Conclusion")
        st.info(get_overall_comment(avg_score, avg_tempo))
        st.caption("This conclusion is automatically generated based on data metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendation Box
    with col_recom:
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("### Key Development Areas")
        
        # Recommendations
        if avg_score < 3.0:
            st.warning("* **Content Development:** Focus on deepening answers according to the rubric. Use the *STAR method* for structure.")
        if avg_tempo > 150 or avg_tempo < 125:
            st.warning("* **Tempo Control:** Practice speaking within the 125-150 BPM range. Practice breathing exercises.")
        if total_pause > 120: 
            st.warning("* **Pause Management:** Reduce excessively long pauses. Consider using short pauses (2-3 seconds) for emphasis only.")
        if avg_confidence < 90:
            st.warning("* **Vocal Clarity Improvement:** Speak louder and clearer. The recording environment should be minimally noisy.")
        
        if avg_score >= 3.5 and 125 <= avg_tempo <= 150 and avg_confidence >= 90:
             st.success("**Excellent Performance:** Your scores are consistently high across all metrics.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- 4. Action Buttons ---
    
    with st.expander("View Detailed Report Per Question"):
        render_detailed_results_per_question() 

    if st.button("Start New Interview 🔄", use_container_width=True, type="primary"):
        st.session_state.clear() 
        next_page('home')

def render_detailed_results_per_question():
    """Function to display results per question, nested within the expander (in English)."""
    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        
        st.markdown(f"### ❓ Question {q_num} Detail: {res['question']}")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Content Score", f"{res['final_score']} / 4")
        with col_res2:
            st.metric("Transcript Accuracy", res['confidence_score'])
        with col_res3:
            st.metric("Analysis Non-Verbal", res['non_verbal'].get('qualitative_summary', 'N/A').capitalize())
        
        st.markdown("**Reason:**")
        st.caption(res['rubric_reason'])

        st.markdown("**Detailed Audio Analysis:**")
        st.markdown(f"* **Tempo:** {res['non_verbal'].get('tempo_bpm', 'N/A')}")
        st.markdown(f"* **Total Pause :** {res['non_verbal'].get('total_pause_seconds', 'N/A')}")

        with st.expander("View Clean Transcript"):
            st.code(res['transcript'], language='text')
        
        st.markdown("---")


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