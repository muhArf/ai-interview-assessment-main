import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np

# =========================
# PATH SETUP
# =========================
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

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
    st.error(f"Failed to load models: {e}")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SEI-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# SESSION STATE
# =========================
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

def next_page(page):
    st.session_state.page = page
    st.rerun()

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def get_models():
    stt = load_stt_model()
    embedder = load_embedder_model()
    spell, _, english_words = load_text_models()
    return stt, embedder, spell, english_words

STT_MODEL, EMBEDDER_MODEL, SPELL_CHECKER, ENGLISH_WORDS = get_models()

@st.cache_data
def load_questions():
    with open("questions.json") as f:
        return json.load(f)

@st.cache_data
def load_rubric_data():
    with open("rubric_data.json") as f:
        return json.load(f)

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()

# =========================
# CSS
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
    #MainMenu, footer, header {visibility:hidden;}

    .metric-grid-container {
        display: flex;
        flex-direction: row;
        gap: 20px;
        width: 100%;
        margin-bottom: 30px;
    }

    .modern-metric-card {
        flex: 1;
        background: #ffffff;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }

    .card-value {
        font-size: 30px;
        font-weight: 700;
        display: block;
    }

    .card-label {
        font-size: 14px;
        color: #7f8c8d;
    }

    .score-color { color:#2ecc71; }
    .accuracy-color { color:#3498db; }
    .tempo-color { color:#f39c12; }
    .pause-color { color:#e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# INTERVIEW PAGE
# =========================
def render_interview_page():
    q_num = st.session_state.current_q
    q_id = str(q_num)

    st.title(f"Interview Question {q_num} of {TOTAL_QUESTIONS}")
    st.info(QUESTIONS[q_id]["question"])

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload your answer video",
            type=["mp4", "mov", "webm"],
            key=f"upload_{q_id}"
        )
        if uploaded:
            st.session_state.answers[q_id] = uploaded
            st.video(uploaded)

    with col2:
        if q_num > 1:
            if st.button("⬅ Previous"):
                st.session_state.current_q -= 1
                st.rerun()

        if q_num < TOTAL_QUESTIONS:
            if st.button("Next ➡", disabled=q_id not in st.session_state.answers):
                st.session_state.current_q += 1
                st.rerun()
        else:
            if st.button("Finish & Process ▶", disabled=q_id not in st.session_state.answers):
                next_page("processing")

# =========================
# PROCESSING PAGE
# =========================
def render_processing_page():
    st.title("Processing Answers...")
    progress = st.progress(0)

    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, TOTAL_QUESTIONS + 1):
            qid = f"q{i}"
            video = st.session_state.answers[str(i)]
            qtext = QUESTIONS[str(i)]["question"]

            video_path = os.path.join(tmp, f"{qid}.mp4")
            audio_path = os.path.join(tmp, f"{qid}.wav")

            with open(video_path, "wb") as f:
                f.write(video.getbuffer())

            video_to_wav(video_path, audio_path)
            noise_reduction(audio_path, audio_path)

            transcript, logprob = transcribe_and_clean(
                audio_path,
                STT_MODEL,
                SPELL_CHECKER,
                EMBEDDER_MODEL,
                ENGLISH_WORDS
            )

            conf = compute_confidence_score(transcript, logprob)
            nonverbal = analyze_non_verbal(audio_path)

            score, reason = score_with_rubric(
                qid,
                qtext,
                transcript,
                RUBRIC_DATA,
                EMBEDDER_MODEL
            )

            results[qid] = {
                "question": qtext,
                "transcript": transcript,
                "final_score": int(score),
                "rubric_reason": reason,
                "confidence_score": f"{conf*100:.2f}",
                "non_verbal": nonverbal
            }

            progress.progress(i / TOTAL_QUESTIONS)

    st.session_state.results = results
    next_page("final_summary")

# =========================
# FINAL SUMMARY PAGE
# =========================
def render_final_summary_page():
    inject_custom_css()
    st.title("🏆 Final Evaluation Report")
    st.markdown("---")

    res = st.session_state.results

    scores = [v["final_score"] for v in res.values()]
    confs = [float(v["confidence_score"]) for v in res.values()]
    tempos = [float(v["non_verbal"]["tempo_bpm"]) for v in res.values()]
    pauses = [float(v["non_verbal"]["total_pause_seconds"]) for v in res.values()]

    avg_score = np.mean(scores)
    avg_conf = np.mean(confs)
    avg_tempo = np.mean(tempos)
    total_pause = np.sum(pauses)

    # ===== Performance Summary =====
    st.subheader("📊 Performance Summary")
    st.markdown('<div class="metric-grid-container">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="modern-metric-card">
        <span class="card-value score-color">🎯 {avg_score:.2f} / 4</span>
        <span class="card-label">Average Content Score</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="modern-metric-card">
        <span class="card-value accuracy-color">🤖 {avg_conf:.2f}%</span>
        <span class="card-label">Transcript Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="modern-metric-card">
        <span class="card-value tempo-color">⏱ {avg_tempo:.2f} BPM</span>
        <span class="card-label">Average Tempo</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="modern-metric-card">
        <span class="card-value pause-color">⏸ {total_pause:.2f}s</span>
        <span class="card-label">Total Pause Time</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ===== Objective Evaluation =====
    st.subheader("💡 Objective Evaluation & Action Plan")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Performance Conclusion")
        st.markdown(
            f"<p style='font-size:15px; line-height:1.7'>"
            f"Overall content relevance and delivery quality are evaluated based on semantic score and communication metrics."
            f"</p>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Key Development Areas")
        recs = []
        if avg_score < 3:
            recs.append("• Improve answer depth using STAR method.")
        if avg_tempo < 125 or avg_tempo > 150:
            recs.append("• Maintain speaking tempo at 125–150 BPM.")
        if total_pause > 120:
            recs.append("• Reduce excessive pauses.")
        if avg_conf < 90:
            recs.append("• Improve articulation and recording clarity.")
        if not recs:
            recs.append("• Excellent overall performance.")

        st.markdown("<br>".join(recs), unsafe_allow_html=True)

    if st.button("Start New Interview 🔄", use_container_width=True):
        st.session_state.clear()
        next_page("home")

# =========================
# ROUTER
# =========================
if st.session_state.page == "interview":
    render_interview_page()
elif st.session_state.page == "processing":
    render_processing_page()
elif st.session_state.page == "final_summary":
    render_final_summary_page()
else:
    st.title("SEI-AI Interviewer")
    if st.button("Start Interview"):
        next_page("interview")
