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

# --- CSS Custom untuk Semua Halaman ---
def inject_global_css():
    """Menyuntikkan CSS kustom untuk semua halaman."""
    st.markdown("""
    <style>
    /* 1. RESET GLOBAL STREAMLIT */
    .stApp {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
        overflow-x: hidden !important;
    }
    
    /* Sembunyikan elemen Streamlit default */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    
    /* 2. NAVBAR CUSTOM */
    .custom-header {
        background-color: white;
        padding: 0 50px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 1000;
        width: 100%;
    }
    
    .navbar-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    
    .navbar-brand {
        font-size: 24px;
        font-weight: 700;
        color: #000;
        text-decoration: none;
    }
    
    .nav-links {
        display: flex;
        gap: 20px;
        align-items: center;
    }
    
    .nav-button {
        background: transparent !important;
        color: #000 !important;
        border: 2px solid #000 !important;
        border-radius: 20px !important;
        padding: 8px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .nav-button:hover {
        background: #000 !important;
        color: white !important;
        transform: translateY(-2px);
    }
    
    /* 3. LANDING PAGE SPECIFIC */
    .hero-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 100px 50px;
        text-align: center;
        border-radius: 0 0 30px 30px;
        margin-bottom: 50px;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 20px;
        color: #000;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #5d5988;
        max-width: 600px;
        margin: 0 auto 40px auto;
        line-height: 1.6;
    }
    
    .primary-btn {
        background: #000 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 15px 40px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .primary-btn:hover {
        background: #333 !important;
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* 4. HOW IT WORKS SECTION */
    .section-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 50px;
        color: #000;
    }
    
    .steps-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        padding: 0 20px;
    }
    
    .step-card {
        background: white;
        border-radius: 15px;
        padding: 40px 25px 25px 25px;
        text-align: center;
        width: 250px;
        min-height: 280px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        position: relative;
        transition: transform 0.3s ease;
    }
    
    .step-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    
    .step-number {
        position: absolute;
        top: -25px;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 50px;
        background: #000;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .step-title {
        font-size: 20px;
        font-weight: 600;
        margin: 20px 0 15px 0;
        color: #000;
    }
    
    .step-description {
        color: #666;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* 5. FOOTER */
    .custom-footer {
        background: #000;
        color: white;
        padding: 30px 50px;
        margin-top: 80px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer-brand {
        font-size: 20px;
        font-weight: 700;
    }
    
    .footer-copyright {
        font-size: 14px;
        opacity: 0.8;
    }
    
    /* 6. METRIC CARDS */
    .metric-grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 20px 0 40px 0;
    }
    
    .modern-metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    
    .card-value {
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .card-label {
        font-size: 13px;
        color: #666;
        margin-top: 5px;
    }
    
    .score-color { color: #2ecc71; }
    .accuracy-color { color: #3498db; }
    .tempo-color { color: #f39c12; }
    .pause-color { color: #e74c3c; }
    
    /* 7. RESPONSIVE */
    @media (max-width: 1200px) {
        .metric-grid-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px;
        }
        
        .hero-section {
            padding: 60px 20px;
        }
        
        .custom-header {
            padding: 0 20px;
        }
        
        .steps-container {
            flex-direction: column;
            align-items: center;
        }
        
        .step-card {
            width: 100%;
            max-width: 300px;
        }
        
        .metric-grid-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Render Functions ---

def render_navbar():
    """Render navbar untuk semua halaman."""
    st.markdown("""
    <div class="custom-header">
        <div class="navbar-content">
            <div class="navbar-brand">SEI-AI</div>
            <div class="nav-links">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🏠 Home", key="nav_home"):
            next_page('home')
    with col2:
        if st.button("ℹ️ Info", key="nav_info"):
            next_page('info')
    
    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_home_page():
    """Render landing page yang sudah diperbaiki."""
    inject_global_css()
    render_navbar()
    
    # HERO SECTION
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="hero-title">AI Interview Assessment Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Latih kemampuan interview Anda dengan AI. Dapatkan evaluasi objektif, analisis jawaban, dan rekomendasi perbaikan instan.</p>', unsafe_allow_html=True)
    
    if st.button("🚀 Mulai Interview Sekarang", key="hero_start", type="primary"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page("interview")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # HOW IT WORKS SECTION
    st.markdown('<h2 class="section-title">Cara Kerja</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    
    steps = [
        ("1", "Unggah Video", "Upload video jawaban interview sesuai pertanyaan yang diberikan"),
        ("2", "Analisis AI", "Sistem menganalisis konten, suara, dan aspek non-verbal"),
        ("3", "Evaluasi Semantik", "Jawaban dinilai berdasarkan relevansi dengan rubrik ideal"),
        ("4", "Dapatkan Feedback", "Terima skor performa dan saran peningkatan"),
        ("5", "Tingkatkan Skill", "Gunakan rekomendasi untuk berlatih lebih baik")
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
    
    # FEATURES SECTION
    st.markdown('<h2 class="section-title" style="margin-top: 80px;">Fitur Unggulan</h2>', unsafe_allow_html=True)
    
    features_cols = st.columns(3)
    with features_cols[0]:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px;">
            <h3>🎤 Speech-to-Text</h3>
            <p>Konversi suara ke teks dengan akurasi tinggi menggunakan AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px;">
            <h3>📊 Analisis Lengkap</h3>
            <p>Evaluasi konten, struktur, dan aspek non-verbal sekaligus</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_cols[2]:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px;">
            <h3>⚡ Feedback Instan</h3>
            <p>Hasil evaluasi dan rekomendasi tersedia dalam hitungan menit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("""
    <div class="custom-footer">
        <div class="footer-brand">SEI-AI Interviewer</div>
        <div class="footer-copyright">© 2024 SEI-AI Interviewer. All Rights Reserved.</div>
    </div>
    """, unsafe_allow_html=True)

def render_info_page():
    """Render halaman informasi."""
    inject_global_css()
    render_navbar()
    
    st.title("📚 Tentang SEI-AI")
    
    st.markdown("""
    ### Teknologi yang Digunakan
    
    Aplikasi ini menggunakan teknologi Machine Learning dan Natural Language Processing (NLP) mutakhir 
    untuk menganalisis jawaban video interview.
    
    #### Proses Analisis
    1. **Speech-to-Text (STT)**: Mengkonversi suara menjadi teks menggunakan model Whisper
    2. **Pembersihan Teks**: Koreksi ejaan dan ambiguitas pada transkrip
    3. **Analisis Non-Verbal**: Analisis tempo bicara, jeda, dan karakteristik suara
    4. **Penilaian Semantik**: Perbandingan jawaban dengan rubrik ideal menggunakan model Sentence-Transformer
    
    #### Persyaratan Video
    * **Durasi**: Disarankan 30-90 detik per jawaban
    * **Format**: MP4, MOV, atau WebM
    * **Ukuran Maksimum**: 50MB
    * **Kualitas Audio**: Pastikan suara jelas dan minim noise
    
    #### Kompatibilitas
    * **Browser**: Chrome, Firefox, Safari terbaru
    * **Device**: Desktop, Tablet, Smartphone
    * **Sistem Operasi**: Windows, macOS, Linux, Android, iOS
    
    ### Keamanan Data
    * Semua video diproses secara real-time
    * Tidak ada data yang disimpan permanen
    * Proses analisis dilakukan secara lokal (jika memungkinkan)
    
    ### Dukungan
    Untuk bantuan teknis atau pertanyaan:
    * Email: support@sei-ai.com
    * Telepon: +62 21 1234 5678
    * Jam Operasional: Senin-Jumat, 09:00-17:00 WIB
    """)
    
    if st.button("🏠 Kembali ke Home", type="primary"):
        next_page('home')

def render_interview_page():
    """Render halaman interview."""
    inject_global_css()
    render_navbar()
    
    st.title(f"🎯 Pertanyaan Interview {st.session_state.current_q} dari {TOTAL_QUESTIONS}")
    
    q_num = st.session_state.current_q
    q_id_str = str(q_num) 
    
    question_data = QUESTIONS.get(q_id_str, {})
    question_text = question_data.get('question', 'Pertanyaan tidak ditemukan.')
    
    if question_text == 'Pertanyaan tidak ditemukan.':
        st.error("Terjadi kesalahan saat memuat pertanyaan.")
        if st.button("🏠 Kembali ke Home"):
            st.session_state.clear() 
            next_page('home')
        return

    st.markdown("### 📝 Pertanyaan:")
    st.info(f"**{question_text}**")
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file_data = st.session_state.answers.get(q_id_str)
    
    # --- Logic untuk Mengelola File Upload ---
    with col_upload:
        uploaded_file = None
        
        if current_uploaded_file_data is None:
            uploaded_file = st.file_uploader(
                f"📤 Unggah Video Jawaban untuk Pertanyaan {q_num} (Maks {VIDEO_MAX_SIZE_MB}MB)",
                type=['mp4', 'mov', 'webm'],
                key=f"uploader_{q_id_str}"
            )
            
            if uploaded_file:
                if uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
                    st.error(f"Ukuran file melebihi batas {VIDEO_MAX_SIZE_MB}MB.")
                else:
                    st.session_state.answers[q_id_str] = uploaded_file
                    current_uploaded_file_data = uploaded_file
                    st.success("✅ File berhasil diunggah!")
                    st.rerun()
        
        if current_uploaded_file_data:
            st.video(current_uploaded_file_data, format=current_uploaded_file_data.type)
            st.info(f"Video jawaban Q{q_num}: **{current_uploaded_file_data.name}**")
            
            if st.button("🗑️ Hapus Video", key=f"delete_q{q_num}", type="secondary"):
                if q_id_str in st.session_state.answers:
                    del st.session_state.answers[q_id_str]
                if f"uploader_{q_id_str}" in st.session_state:
                    del st.session_state[f"uploader_{q_id_str}"]
                st.rerun()
        else:
            st.warning("Silakan unggah video jawaban untuk melanjutkan.")
    
    # --- Kontrol Navigasi ---
    with col_control:
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("⏭️ Pertanyaan Selanjutnya", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("🏁 Selesai & Proses", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')
        
        if q_num > 1:
            if st.button("⏮️ Sebelumnya", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

def render_processing_page():
    """Render halaman processing."""
    inject_global_css()
    render_navbar()
    
    st.title("⚙️ Proses Analisis")
    st.info("Harap tunggu, proses ini membutuhkan beberapa menit tergantung durasi video.")
    
    if st.session_state.results is not None and st.session_state.results != {}:
        next_page('final_summary') 
        return
    
    if st.session_state.results is None:
        results = {}
        progress_bar = st.progress(0, text="Memulai proses...")
        
        if not all([STT_MODEL, EMBEDDER_MODEL]):
            st.error("Model utama gagal dimuat. Tidak dapat melanjutkan proses.")
            progress_bar.empty()
            if st.button("🏠 Kembali ke Home"):
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
                        st.markdown(f"### Memproses Q{i}: {q_text[:50]}...")
                        
                        progress_bar.progress((i-1)*10 + 1, text=f"Q{i}: Menyimpan video...")
                        temp_video_path = os.path.join(temp_dir, f'video_{q_key_rubric}.mp4')
                        temp_audio_path = os.path.join(temp_dir, f'audio_{q_key_rubric}.wav')
                        
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_file.getbuffer())
                        
                        progress_bar.progress((i-1)*10 + 3, text=f"Q{i}: Ekstraksi audio...")
                        video_to_wav(temp_video_path, temp_audio_path)
                        noise_reduction(temp_audio_path, temp_audio_path)
                        
                        progress_bar.progress((i-1)*10 + 5, text=f"Q{i}: Transkripsi...")
                        transcript, log_prob_raw = transcribe_and_clean(
                            temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        final_confidence_score = compute_confidence_score(transcript, log_prob_raw)
                        
                        progress_bar.progress((i-1)*10 + 7, text=f"Q{i}: Analisis non-verbal...")
                        non_verbal_res = analyze_non_verbal(temp_audio_path)
                        
                        progress_bar.progress((i-1)*10 + 9, text=f"Q{i}: Penilaian semantik...")
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[ERROR: Gagal menghitung skor] {reason}"
                        
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value,
                            "rubric_reason": reason,
                            "confidence_score": f"{final_confidence_score*100:.2f}",
                            "non_verbal": non_verbal_res
                        }
                        
                        progress_bar.progress(i*10, text=f"Q{i} Selesai.")
                    else:
                        st.warning(f"Melewati Q{i}: File tidak ditemukan atau data rubrik tidak ada.")
                
                st.session_state.results = results
                progress_bar.progress(100, text="Proses selesai! Mengarahkan ke laporan...")
                next_page('final_summary')
        
        except Exception as e:
            st.error(f"Kesalahan fatal selama proses: {e}")
            st.warning("Proses dibatalkan. Silakan coba kembali.")
            progress_bar.empty()
            st.session_state.results = None
            if st.button("🏠 Kembali ke Home"):
                st.session_state.clear()
                next_page('home')
            return

def render_final_summary_page():
    """Render halaman hasil akhir."""
    inject_global_css()
    render_navbar()
    
    st.title("🏆 Laporan Evaluasi Final")
    st.markdown("---")
    
    if not st.session_state.results:
        st.error("Data hasil tidak ditemukan.")
        if st.button("Kembali"):
            next_page('home')
        return
    
    # Hitung metrik
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
        st.error(f"Gagal menghitung metrik: {e}")
        return
    
    # Tampilkan metrik
    st.subheader("📊 Ringkasan Performa")
    
    st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-value score-color">{avg_score:.2f}/4</div>
        <div class="card-label">Skor Konten Rata-rata</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-value accuracy-color">{avg_confidence:.2f}%</div>
        <div class="card-label">Akurasi Transkripsi</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-value tempo-color">{avg_tempo:.1f}</div>
        <div class="card-label">Tempo Bicara (BPM)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="modern-metric-card">
        <div class="card-value pause-color">{total_pause:.1f}s</div>
        <div class="card-label">Total Jeda</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Evaluasi dan Rekomendasi
    st.markdown("---")
    st.subheader("💡 Evaluasi & Rekomendasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Kesimpulan Performa")
        if avg_score >= 3.5:
            st.success("**Excellent!** Konten jawaban sangat relevan dan terstruktur dengan baik.")
        elif avg_score >= 2.5:
            st.warning("**Good.** Konten cukup baik namun bisa ditingkatkan dengan struktur yang lebih jelas.")
        else:
            st.error("**Needs Improvement.** Perlu peningkatan pada relevansi jawaban dengan pertanyaan.")
        
        if 125 <= avg_tempo <= 150:
            st.success("**Tempo bicara optimal** untuk komunikasi efektif.")
        elif avg_tempo > 150:
            st.warning("**Tempo terlalu cepat.** Cobalah berbicara lebih pelan untuk kejelasan.")
        else:
            st.warning("**Tempo terlalu lambat.** Tingkatkan kecepatan bicara untuk menjaga perhatian.")
    
    with col2:
        st.markdown("### Area Pengembangan")
        if avg_score < 3.0:
            st.info("""
            **🎯 Fokus Konten:**
            - Gunakan metode STAR (Situation, Task, Action, Result)
            - Sertakan contoh spesifik dan data pendukung
            - Sesuaikan dengan rubrik penilaian
            """)
        
        if total_pause > 120:
            st.info("""
            **⏸️ Manajemen Jeda:**
            - Kurangi jeda yang terlalu panjang
            - Gunakan jeda 2-3 detik untuk penekanan
            - Latih ritme bicara yang konsisten
            """)
        
        if avg_confidence < 90:
            st.info("""
            **🎤 Kejelasan Suara:**
            - Tingkatkan volume dan artikulasi
            - Pilih lingkungan rekaman yang tenang
            - Gunakan mikrofon eksternal jika memungkinkan
            """)
    
    # Detail per pertanyaan
    st.markdown("---")
    with st.expander("📋 Lihat Detail per Pertanyaan"):
        for q_key, res in st.session_state.results.items():
            q_num = q_key.replace('q', '')
            
            st.markdown(f"### Pertanyaan {q_num}")
            st.write(f"**Pertanyaan:** {res['question']}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Skor Konten", f"{res['final_score']}/4")
            with col_b:
                st.metric("Akurasi", f"{res['confidence_score']}%")
            with col_c:
                st.metric("Analisis Suara", res['non_verbal'].get('qualitative_summary', 'N/A'))
            
            st.markdown("**Penilaian:**")
            st.info(res['rubric_reason'])
            
            with st.expander("Lihat Transkrip"):
                st.code(res['transcript'], language='text')
            
            st.markdown("---")
    
    # Tombol aksi
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔄 Interview Baru", use_container_width=True):
            st.session_state.clear()
            next_page('home')
    
    with col_btn2:
        if st.button("📥 Download Laporan", use_container_width=True):
            st.info("Fitur download akan segera tersedia!")
    
    with col_btn3:
        if st.button("🏠 Kembali ke Home", use_container_width=True):
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
elif st.session_state.page == 'final_summary':
    render_final_summary_page()