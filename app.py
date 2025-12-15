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
    from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
    from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
    from models.nonverbal_analysis import analyze_non_verbal
except ImportError as e:
    # Handle the error gracefully if modules fail to load
    st.error(f"Gagal memuat modul dari folder 'models'. Pastikan struktur folder dan file sudah benar. Error: {e}")
    # Jika Anda ingin aplikasi berhenti total di sini, gunakan:
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
        stt_model = load_stt_model()
        embedder_model = load_embedder_model()
        
        try:
            spell, _, english_words = load_text_models()
        except Exception:
            spell, english_words = None, None
            st.warning("Gagal memuat model spell checker/kata-kata. Pembersihan teks akan terbatas.")
            
        return stt_model, embedder_model, spell, english_words
    except Exception as e:
        st.error(f"Gagal memuat salah satu model inti. Pastikan semua dependensi terinstal. Error: {e}")
        return None, None, None, None

# Load model di awal
STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

@st.cache_data
def load_questions():
    """Memuat pertanyaan dari questions.json."""
    try:
        with open('questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File questions.json tidak ditemukan! Pastikan file ada.")
        return {}

@st.cache_data
def load_rubric_data():
    """Memuat data rubrik dari rubric_data.json."""
    try:
        with open('rubric_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File rubric_data.json tidak ditemukan! Pastikan file ada.")
        return {}

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()

# --- Page Render Functions ---

# --- LANDING PAGE BARU (START) ---

def inject_custom_css():
    """Menyuntikkan CSS kustom untuk meniru desain Landing Page."""
    st.markdown("""
    <style>
    /* 1. Reset Global dan Kontrol Padding */
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
    
    /* Memaksa elemen di kolom nav Streamlit untuk rata kanan */
    /* Ini adalah perbaikan utama untuk memastikan tombol-tombol sejajar horizontal */
    .header-nav > div[data-testid="stHorizontalBlock"] {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        width: 100%;
    }
    .header-nav > div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
        /* Kolom Home */
        margin-right: 20px;
    }
    .header-nav > div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        /* Kolom Info Aplikasi */
        margin-right: 10px;
    }
    .header-nav button {
        /* Menyeimbangkan posisi tombol agar sejajar dengan teks Home */
        margin-top: 0px !important;
        padding: 8px 15px !important;
        font-size: 14px !important;
        border-radius: 6px !important;
        height: 40px; 
    }

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

    /* Styling How To Use Steps (Diperkuat) */
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
        min-height: 250px; /* Menjaga tinggi tetap untuk kerapian */
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
    }

    /* Penyesuaian Tombol Hero Streamlit */
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
                st.image('assets/logo dicoding.png', width=80, output_format='PNG') 
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
                if st.button("Info Aplikasi", key="nav_info", type="secondary"):
                    next_page('info')
            
            with col_start:
                # Tombol Mulai Wawancara di Navbar
                if st.button("Mulai Wawancara", key="nav_start", type="primary"):
                    st.session_state.answers = {}
                    st.session_state.results = None
                    st.session_state.current_q = 1
                    next_page('interview')
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # --- 2. Hero Section ---
    st.markdown('<section class="hero-section">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="hero-title">Selamat Datang di SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Asah keterampilan wawancara Anda dengan umpan balik bertenaga AI dan bersiaplah untuk pekerjaan impian Anda.</p>', unsafe_allow_html=True)
    
    # Tombol Aksi Hero Section
    st.markdown('<div class="primary-btn-container" style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    if st.button("▶️ Mulai Wawancara", key="hero_start"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page('interview')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</section>', unsafe_allow_html=True)


    # --- 3. How To Use Section (Langkah-Langkah) ---
    st.markdown('<section style="padding: 50px; background-color: white;">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 40px; text-align: center; margin-bottom: 70px;">Bagaimana Cara Menggunakan</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    
    # Grid Langkah-Langkah: Menggunakan kolom Streamlit untuk 5 card
    cols = st.columns(5)
    
    steps_data = [
        ("1", "Unggah Video Jawaban", "Unggah video jawaban Anda untuk setiap pertanyaan yang diberikan AI."),
        ("2", "AI Memproses Data", "AI akan memproses video menjadi transkrip dan menganalisis aspek non-verbal."),
        ("3", "Penilaian Semantik", "Jawaban Anda dibandingkan dengan rubrik ideal untuk penilaian semantik (Relevansi konten)."),
        ("4", "Dapatkan Umpan Balik", "Terima skor akhir, alasan, dan analisis komunikasi instan."),
        ("5", "Tingkatkan Keterampilan", "Gunakan rekomendasi untuk meningkatkan dan berlatih hingga Anda percaya diri.")
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

# --- LANDING PAGE BARU (END) ---


def render_info_page():
    st.title("Informasi Aplikasi SEI-AI")
    st.markdown("""
    Aplikasi ini menggunakan teknologi Machine Learning dan Natural Language Processing (NLP) untuk menganalisis jawaban wawancara video.
    
    ### Proses Analisis
    1. **Speech-to-Text (STT):** Transkrip jawaban Anda dibuat menggunakan model *Whisper*.
    2. **Pembersihan Teks:** Teks transkrip dikoreksi dari kesalahan ejaan (spell check) dan ketidakjelasan.
    3. **Analisis Non-Verbal:** Audio dianalisis untuk Tempo Bicara (BPM) dan total Jeda (Pause).
    4. **Penilaian Semantik:** Jawaban dinilai berdasarkan perbandingan semantik dengan poin-poin rubrik ideal menggunakan model *Sentence-Transformer*.
    
    ### Persyaratan Unggahan Video
    * **Durasi:** Direkomendasikan 30-60 detik per jawaban.
    * **Format:** MP4, MOV, atau WebM.
    * **Ukuran Maksimal:** 50MB.
    """)
    if st.button("🏠 Kembali ke Awal"):
        next_page('home')

def render_interview_page():
    st.title(f"Pertanyaan Wawancara Ke-{st.session_state.current_q} dari {TOTAL_QUESTIONS}")
    
    q_num = st.session_state.current_q
    q_id_str = str(q_num) 
    
    question_data = QUESTIONS.get(q_id_str, {})
    question_text = question_data.get('question', 'Pertanyaan tidak ditemukan.')
    
    if question_text == 'Pertanyaan tidak ditemukan.':
        st.error("Terjadi kesalahan saat memuat pertanyaan.")
        if st.button("🏠 Kembali ke Awal"):
            st.session_state.clear() 
            next_page('home')
        return

    st.header("Pertanyaan:")
    st.info(question_text)
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_uploaded_file = st.session_state.answers.get(q_id_str)

    with col_upload:
        uploaded_file = st.file_uploader(
            f"Upload Video Jawaban untuk Pertanyaan {q_num} (Max {VIDEO_MAX_SIZE_MB}MB)",
            type=['mp4', 'mov', 'webm'],
            key=f"uploader_{q_id_str}"
        )

        if uploaded_file and uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
            st.error(f"Ukuran file melebihi batas {VIDEO_MAX_SIZE_MB}MB. File tidak akan diproses.")
            uploaded_file = None
        
        st.session_state.answers[q_id_str] = uploaded_file

        if uploaded_file:
            st.success("File berhasil diunggah.")
            st.video(uploaded_file, format=uploaded_file.type)
        elif current_uploaded_file:
            st.video(current_uploaded_file, format=current_uploaded_file.type)
            st.info("File sebelumnya terdeteksi.")
        else:
            st.warning("Silakan unggah file jawaban Anda untuk melanjutkan.")

    with col_control:
        st.markdown("### Kontrol")
        
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("Pertanyaan Selanjutnya ⏩", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("Selesai & Proses ▶️", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')

        if q_num > 1:
            if st.button("⏪ Pertanyaan Sebelumnya", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

def render_processing_page():
    st.title("⚙️ Proses Analisis Jawaban")
    st.info("Harap tunggu, proses ini mungkin memakan waktu beberapa menit tergantung durasi video.")

    if st.session_state.results is not None and st.session_state.results != {}:
        next_page('results')
        return

    if st.session_state.results is None:
        
        results = {}
        progress_bar = st.progress(0, text="Memulai Proses...")
        
        if not all([STT_MODEL, EMBEDDER_MODEL]):
            st.error("Model inti gagal dimuat. Tidak dapat melanjutkan pemrosesan.")
            progress_bar.empty()
            if st.button("🏠 Kembali ke Awal"):
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
                        
                        # --- 1. Simpan Video 
                        progress_bar.progress((i-1)*10 + 1, text=f"Q{i}: Menyimpan video...")
                        temp_video_path = os.path.join(temp_dir, f'video_{q_key_rubric}.mp4')
                        with open(temp_video_path, 'wb') as f:
                            f.write(video_file.getbuffer())

                        # --- 2. Ekstraksi Audio & Noise Reduction
                        progress_bar.progress((i-1)*10 + 3, text=f"Q{i}: Ekstraksi audio dan Noise Reduction...")
                        temp_audio_path = os.path.join(temp_dir, f'audio_{q_key_rubric}.wav')
                        video_to_wav(temp_video_path, temp_audio_path)
                        
                        noise_reduction(temp_audio_path, temp_audio_path) 
                        
                        # --- 3. Speech-to-Text (STT) & Cleaning
                        progress_bar.progress((i-1)*10 + 5, text=f"Q{i}: Transkripsi dan Pembersihan Teks...")
                        transcript, log_prob_raw = transcribe_and_clean(
                            temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                        )
                        
                        final_confidence_score_0_1 = compute_confidence_score(transcript, log_prob_raw)
                        
                        # --- 4. Analisis Non-Verbal
                        progress_bar.progress((i-1)*10 + 7, text=f"Q{i}: Analisis Non-Verbal...")
                        non_verbal_res = analyze_non_verbal(temp_audio_path)

                        # --- 5. Penilaian Jawaban (Semantik)
                        progress_bar.progress((i-1)*10 + 9, text=f"Q{i}: Penilaian Semantik...")
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[ERROR: Skor gagal dihitung/Tipe data salah. Skor default 0 digunakan.] {reason}"
                        
                        # --- 6. Simpan Hasil
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value,
                            "rubric_reason": reason,
                            "confidence_score": f"{final_confidence_score_0_1*100:.2f}",
                            "non_verbal": non_verbal_res
                        }

                        progress_bar.progress(i*10, text=f"Q{i} Selesai.")
                    else:
                        st.warning(f"Melewati Q{i} (ID: {q_id_str}): File jawaban tidak diunggah atau data rubrik hilang.")
                    
                st.session_state.results = results
                progress_bar.progress(100, text="Proses Selesai! Menuju Halaman Hasil.")
                next_page('results')

        except Exception as e:
            st.error(f"Terjadi kesalahan fatal selama pemrosesan: {e}")
            st.warning("Pemrosesan dibatalkan. Silakan coba kembali ke awal.")
            progress_bar.empty()
            st.session_state.results = None 
            if st.button("🏠 Kembali ke Awal"):
                 st.session_state.clear() 
                 next_page('home')
            return

def render_results_page():
    # Fungsi ini sekarang hanya bertindak sebagai pengalih (redirector)
    # untuk memastikan pengguna langsung melihat Laporan Akumulasi Akhir.
    
    if not st.session_state.results:
        st.error("Hasil tidak ditemukan. Silakan coba proses ulang.")
        if st.button("🏠 Kembali ke Awal"):
            st.session_state.clear()
            next_page('home')
        return
        
    # --- PENGALIHAN LANGSUNG KE LAPORAN AKUMULASI AKHIR ---
    next_page('final_summary')

    # --- SKOR TOTAL DIHILANGKAN SESUAI PERMINTAAN ---
    st.markdown("---") 

    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        
        # Header untuk setiap pertanyaan
        st.header(f"Laporan Analisis Pertanyaan {q_num}")
        st.info(f"**Pertanyaan:** {res['question']}")
        
        # --- 1. Key Metrics (Skor dan Rangkuman) ---
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            score_str = str(res['final_score'])
            # Tampilkan skor Kualitas Jawaban
            st.metric("Skor Konten (Maks 4)", f"**{score_str} / 4**")
        with col_res2:
            # Mengganti Confidence Score menjadi Akurasi Transkrip
            confidence_val = float(res['confidence_score'].replace('%', '').replace(' per minute', '').replace(' seconds', ''))
            st.metric("Akurasi Transkrip (STT)", f"{confidence_val:.2f}%")
        with col_res3:
            summary = res['non_verbal'].get('qualitative_summary', 'N/A')
            # Mengganti Rangkuman Non-Verbal menjadi Rangkuman Komunikasi
            st.metric("Rangkuman Komunikasi", summary.capitalize())
        
        st.markdown("<br>", unsafe_allow_html=True) # Tambahkan jarak

        # --- 2. Detail Penilaian Konten (Expanded by default) ---
        with st.expander("📝 Detail Penilaian Konten (Rubrik Semantik)", expanded=True):
            st.subheader("Alasan Penilaian Skor")
            st.write(res['rubric_reason'])

        # --- 3. Detail Analisis Non-Verbal (Collapsed by default) ---
        with st.expander("🗣️ Detail Analisis Non-Verbal (Audio)", expanded=False):
            # Asumsi data tempo dan pause sudah diformat dengan satuan di nonverbal_analysis.py
            tempo = res['non_verbal'].get('tempo_bpm', 'N/A')
            pause = res['non_verbal'].get('total_pause_seconds', 'N/A')
            
            st.markdown(f"* **Tempo Bicara:** {tempo}")
            st.markdown(f"* **Total Jeda (Keheningan):** {pause}")

        # --- 4. Transkrip Jawaban Bersih (Collapsed by default) ---
        with st.expander("📄 Transkrip Jawaban Bersih", expanded=False):
            st.code(res['transcript'], language='text')
        
        # Pemisah tebal antar pertanyaan
        st.markdown("<br><hr style='border: 4px solid #f0f2f6; border-radius: 5px;'>", unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True) 


    if st.button("🏠 Selesai & Kembali ke Awal", use_container_width=True):
        st.session_state.clear() 
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
    render_results_page() # Fungsi ini sekarang hanya pengalih
elif st.session_state.page == 'final_summary':
    render_final_summary_page() # Halaman Laporan Akumulasi Utama