# app.py
import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# Tambahkan direktori saat ini dan 'models' ke PATH
# Ini memastikan impor dari models/* berfungsi
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import logic dari folder models
# PASTIKAN FILE-FILE INI ADA DI FOLDER 'models'
from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
from models.nonverbal_analysis import analyze_non_verbal

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
TOTAL_QUESTIONS = 5 # Jumlah Pertanyaan
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
        
        # Load model text
        try:
            # Perhatikan: load_text_models mengembalikan 3 nilai: spell, model_embedder (tidak dipakai lagi di sini), english_words
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
        # ASUMSI: File questions.json ada di root folder
        with open('questions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File questions.json tidak ditemukan! Pastikan file ada.")
        return {}

@st.cache_data
def load_rubric_data():
    """Memuat data rubrik dari rubric_data.json."""
    try:
        # ASUMSI: File rubric_data.json ada di root folder
        with open('rubric_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File rubric_data.json tidak ditemukan! Pastikan file ada.")
        return {}

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()

# --- Page Render Functions ---

def render_home_page():
    st.title("Selamat Datang di SEI-AI Interviewer")
    st.markdown("Aplikasi ini akan memandu Anda melalui 5 pertanyaan wawancara. Silakan persiapkan video jawaban Anda untuk diunggah.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Mulai Wawancara")
        if st.button("Mulai Sekarang", use_container_width=True):
            # Membersihkan state sesi sebelumnya
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page('interview')
    with col2:
        st.header("Informasi")
        if st.button("Lihat Info Aplikasi", use_container_width=True):
            next_page('info')
            
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
    # q_id_str: Key untuk QUESTIONS (questions.json) dan st.session_state.answers (contoh: "1", "2")
    q_id_str = str(q_num) 
    
    # FIX: Akses QUESTIONS menggunakan key string dari nomor pertanyaan ("1", "2", dst)
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
    
    # Kolom Unggahan Video
    col_upload, col_control = st.columns([3, 1])
    
    # Nilai awal yang sudah diunggah
    current_uploaded_file = st.session_state.answers.get(q_id_str)

    with col_upload:
        uploaded_file = st.file_uploader(
            f"Upload Video Jawaban untuk Pertanyaan {q_num} (Max {VIDEO_MAX_SIZE_MB}MB)",
            type=['mp4', 'mov', 'webm'],
            # FIX: Gunakan q_id_str yang benar ("1", "2", dst) untuk key uploader
            key=f"uploader_{q_id_str}"
        )

        # Cek ukuran file
        if uploaded_file and uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
            st.error(f"Ukuran file melebihi batas {VIDEO_MAX_SIZE_MB}MB. File tidak akan diproses.")
            uploaded_file = None
        
        # FIX: Gunakan q_id_str sebagai key untuk menyimpan file
        st.session_state.answers[q_id_str] = uploaded_file

        if uploaded_file:
            st.success("File berhasil diunggah.")
            st.video(uploaded_file, format=uploaded_file.type)
        elif current_uploaded_file:
            # Tampilkan file yang sudah diunggah dari session state
            st.video(current_uploaded_file, format=current_uploaded_file.type)
            st.info("File sebelumnya terdeteksi.")
        else:
            st.warning("Silakan unggah file jawaban Anda untuk melanjutkan.")

    with col_control:
        st.markdown("### Kontrol")
        
        # Tombol Selanjutnya/Selesai
        is_ready = st.session_state.answers.get(q_id_str) is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("Pertanyaan Selanjutnya ⏩", use_container_width=True, disabled=(not is_ready)):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("Selesai & Proses ▶️", use_container_width=True, disabled=(not is_ready)):
                next_page('processing')

        # Tombol Sebelumnya
        if q_num > 1:
            if st.button("⏪ Pertanyaan Sebelumnya", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

def render_processing_page():
    st.title("⚙️ Proses Analisis Jawaban")
    st.info("Harap tunggu, proses ini mungkin memakan waktu beberapa menit tergantung durasi video.")

    # Jika hasil sudah ada, langsung tampilkan
    if st.session_state.results is not None and st.session_state.results != {}:
        next_page('results')
        return

    if st.session_state.results is None:
        
        results = {}
        progress_bar = st.progress(0, text="Memulai Proses...")
        
        # Cek apakah model inti berhasil dimuat
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
                    # q_id_str: Key untuk answers dan questions.json (e.g., "1")
                    q_id_str = str(i) 
                    # q_key_rubric: Key untuk rubric_data.json dan hasil akhir (e.g., "q1")
                    q_key_rubric = f'q{i}' 

                    # FIX 1: Akses answers menggunakan q_id_str
                    video_file = st.session_state.answers.get(q_id_str)
                    
                    # FIX 2: Akses QUESTIONS menggunakan q_id_str
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
                        
                        # Hitung confidence score akhir (0.0 - 1.0) menggunakan log_prob_raw
                        final_confidence_score_0_1 = compute_confidence_score(transcript, log_prob_raw)
                        
                        # --- 4. Analisis Non-Verbal
                        progress_bar.progress((i-1)*10 + 7, text=f"Q{i}: Analisis Non-Verbal...")
                        non_verbal_res = analyze_non_verbal(temp_audio_path)

                        # --- 5. Penilaian Jawaban (Semantik)
                        progress_bar.progress((i-1)*10 + 9, text=f"Q{i}: Penilaian Semantik...")
                        score, reason = score_with_rubric(
                            q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                        )
                        
                        # PERBAIKAN: Pastikan 'score' adalah integer yang valid.
                        try:
                            final_score_value = int(score) if score is not None else 0
                        except (ValueError, TypeError):
                            final_score_value = 0
                            reason = f"[ERROR: Skor gagal dihitung. Skor default 0 digunakan.] {reason}"
                        
                        # --- 6. Simpan Hasil
                        results[q_key_rubric] = {
                            "question": q_text,
                            "transcript": transcript,
                            "final_score": final_score_value, # Menggunakan nilai yang sudah divalidasi
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
            # FIX 5: Set results ke None saat error agar tidak masuk ke halaman hasil yang rusak
            st.session_state.results = None 
            if st.button("🏠 Kembali ke Awal"):
                 st.session_state.clear() 
                 next_page('home')
            return

def render_results_page():
    st.title("✅ Hasil Analisis Wawancara")
    
    if not st.session_state.results:
        st.error("Hasil tidak ditemukan. Silakan coba proses ulang.")
        if st.button("🏠 Kembali ke Awal"):
            st.session_state.clear()
            next_page('home')
        return

    # Hitung Skor Total
    processed_q = len(st.session_state.results)
    if processed_q > 0:
        # PERBAIKAN: Konversi ke int() untuk memastikan penjumlahan berjalan lancar.
        total_score = sum(int(res['final_score']) for res in st.session_state.results.values())
        max_score = processed_q * 4 
    else:
        total_score = 0
        max_score = 0
    
    st.markdown("---")
    st.header(f"Skor Total Wawancara: **{total_score} / {max_score}**")
    st.markdown("---")

    # Tampilkan hasil per pertanyaan
    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        
        st.subheader(f"Pertanyaan {q_num}: {res['question']}")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Skor Final (Semantik)", f"**{res['final_score']} / 4**", unsafe_allow_html=True)
        with col_res2:
            # Pastikan nilai adalah float sebelum format
            confidence_val = float(res['confidence_score'].replace('%', ''))
            st.metric("Confidence Score", f"{confidence_val:.2f}%")
        with col_res3:
            summary = res['non_verbal'].get('qualitative_summary', 'N/A')
            st.metric("Rangkuman Non-Verbal", summary.capitalize())
        
        st.markdown("---")

        st.markdown("**Rubrik Penilaian (Alasan Semantik)**")
        st.caption(f"**Alasan Pemberian Skor:** {res['rubric_reason']}")

        st.markdown("**Analisis Audio (Non-Verbal Detail)**")
        # Tampilkan detail tempo dan pause
        tempo = res['non_verbal'].get('tempo_bpm', 'N/A')
        pause = res['non_verbal'].get('total_pause_seconds', 'N/A')
        st.markdown(f"* **Tempo Bicara (BPM):** {tempo}")
        st.markdown(f"* **Total Jeda (Detik):** {pause}")

        st.markdown("**Transkrip Jawaban Bersih**")
        st.code(res['transcript'], language='text')
        
        st.markdown("---")


    if st.button("🏠 Selesai & Kembali ke Awal", use_container_width=True):
        # Membersihkan session state untuk memulai ulang aplikasi sepenuhnya
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
    render_results_page()