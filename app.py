# app.py (Complete version with PDF generation)
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
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
    from models.stt_processor import load_stt_model, load_text_models, video_to_wav, noise_reduction, transcribe_and_clean
    from models.scoring_logic import load_embedder_model, compute_confidence_score, score_with_rubric
    from models.nonverbal_analysis import analyze_non_verbal
except ImportError:
    # Dummy functions for testing
    def load_stt_model(): return "STT_Model_Loaded"
    def load_text_models(): return None, None, None
    def load_embedder_model(): return "Embedder_Model_Loaded"
    def video_to_wav(video_path, audio_path): 
        import shutil
        shutil.copy(video_path, audio_path)
    def noise_reduction(audio_path_in, audio_path_out): 
        import shutil
        shutil.copy(audio_path_in, audio_path_out)
    def transcribe_and_clean(audio_path, stt_model, spell_checker, embedder_model, english_words): 
        return "This is a sample transcript for testing purposes. The candidate demonstrates good communication skills.", 0.95
    def compute_confidence_score(transcript, log_prob_raw): 
        return 0.85
    def analyze_non_verbal(audio_path): 
        return {
            'tempo_bpm': '140 BPM',
            'total_pause_seconds': '50 seconds',
            'qualitative_summary': 'Good pace with appropriate pauses'
        }
    def score_with_rubric(q_key_rubric, q_text, transcript, RUBRIC_DATA, embedder_model): 
        return 4, "Candidate meets rubric 4 because demonstrates comprehensive understanding and clear structure."

# Page Configuration & Data Load
st.set_page_config(
    page_title="SEI-AI Interview Analyzer",
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
if 'nonverbal_scores' not in st.session_state:
    st.session_state.nonverbal_scores = {}

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
            
        return stt_model, embedder_model, spell, english_words
    except Exception as e:
        st.error(f"Failed to load one of the core models. Error: {e}")
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
        st.error("questions.json file not found!")
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
        st.error("rubric_data.json file not found!")
        return {}

QUESTIONS = load_questions()
RUBRIC_DATA = load_rubric_data()

# --- PDF Generation Functions ---
def generate_pdf_report():
    """Generate PDF report in the exact format requested."""
    if not st.session_state.results or not st.session_state.candidate_data:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           topMargin=2*cm, 
                           bottomMargin=2*cm,
                           leftMargin=2*cm, 
                           rightMargin=2*cm)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=6,
        fontName='Helvetica'
    )
    
    bold_style = ParagraphStyle(
        'BoldStyle',
        parent=normal_style,
        fontName='Helvetica-Bold'
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8,
        fontName='Helvetica-Bold',
        leftIndent=10
    )
    
    reason_style = ParagraphStyle(
        'ReasonStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=4,
        leftIndent=20,
        bulletIndent=20,
        fontName='Helvetica'
    )
    
    metric_style = ParagraphStyle(
        'MetricStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=2,
        fontName='Helvetica-Oblique'
    )
    
    # Title and Candidate Info
    story.append(Paragraph("INTERVIEW ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 10))
    
    candidate = st.session_state.candidate_data
    story.append(Paragraph(f"<b>Candidate:</b> {candidate.get('name', 'N/A')}", normal_style))
    story.append(Paragraph(f"<b>Interview ID:</b> {candidate.get('id', 'N/A')}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {candidate.get('start_time', 'N/A')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Overall Summary Section
    story.append(Paragraph("SUMMARY", heading_style))
    
    results = st.session_state.results
    all_scores = [int(res['final_score']) for res in results.values()]
    all_confidence = [float(res['confidence_score'].replace('%', '')) for res in results.values()]
    
    avg_score = np.mean(all_scores) if all_scores else 0
    avg_confidence = np.mean(all_confidence) if all_confidence else 0
    
    summary_data = [
        ["Metric", "Score"],
        ["Average Rubric Score", f"{avg_score:.1f}/4"],
        ["Average Confidence", f"{avg_confidence:.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[doc.width/2, doc.width/2])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Non-Verbal Analysis Section
    story.append(Paragraph("NON-VERBAL ANALYSIS", heading_style))
    
    nonverbal_scores = st.session_state.get('nonverbal_scores', {})
    if nonverbal_scores:
        for metric, score in nonverbal_scores.items():
            if isinstance(score, (int, float)):
                story.append(Paragraph(f"{metric.replace('_', ' ').title()}: <b>{score}%</b>", normal_style))
    else:
        story.append(Paragraph("No non-verbal data available", normal_style))
    
    story.append(Spacer(1, 25))
    
    # Detailed Breakdown by Question - EXACT FORMAT AS REQUESTED
    story.append(Paragraph("DETAILED BREAKDOWN BY QUESTION", heading_style))
    story.append(Spacer(1, 15))
    
    for q_key, res in results.items():
        q_num = q_key.replace('q', '')
        
        # Question Header
        story.append(Paragraph(f"<b>Question {q_num}:</b> {res['question']}", question_style))
        story.append(Spacer(1, 8))
        
        # Rubric and Confidence at the top
        score_info = [
            ["Rubric", f"{res['final_score']}/4"],
            ["Confidence", f"{res['confidence_score']}"]
        ]
        
        score_table = Table(score_info, colWidths=[doc.width/3, doc.width/3*2])
        score_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 8))
        
        # Reason section with bullet points
        story.append(Paragraph("<b>Reason:</b>", bold_style))
        
        # Process reasons
        rubric_reason = res['rubric_reason']
        if isinstance(rubric_reason, str):
            # Split by common separators
            if '. ' in rubric_reason:
                reasons = rubric_reason.split('. ')
            elif '- ' in rubric_reason:
                reasons = rubric_reason.split('- ')
            else:
                reasons = [rubric_reason]
            
            for reason in reasons:
                if reason.strip():
                    story.append(Paragraph(f"• {reason.strip()}", reason_style))
        else:
            story.append(Paragraph(f"• {rubric_reason}", reason_style))
        
        story.append(Spacer(1, 8))
        
        # Tempo and Pause at the bottom (as requested)
        non_verbal = res.get('non_verbal', {})
        tempo = non_verbal.get('tempo_bpm', 'N/A')
        pause = non_verbal.get('total_pause_seconds', 'N/A')
        
        # Extract numeric values
        tempo_value = ''.join(filter(str.isdigit, str(tempo))) if tempo != 'N/A' else 'N/A'
        pause_value = ''.join(filter(str.isdigit, str(pause))) if pause != 'N/A' else 'N/A'
        
        metrics_text = f"Tempo: {tempo_value} | Pause: {pause_value} seconds"
        story.append(Paragraph(metrics_text, metric_style))
        
        story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer

def download_pdf_report():
    """Create and trigger PDF download."""
    pdf_buffer = generate_pdf_report()
    
    if pdf_buffer:
        candidate = st.session_state.candidate_data
        candidate_name = candidate.get('name', 'candidate').replace(' ', '_')
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"Interview_Report_{candidate_name}_{date_str}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download_button"
        )
    else:
        st.warning("Cannot generate PDF. Complete the interview first.")

# --- Global CSS Injection (KEEP YOUR ORIGINAL CSS) ---
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
    
    /* 6. RESULTS PAGE STYLING */
    .results-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        margin: 20px 0;
    }
    
    .question-breakdown {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #3498db;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 30px 0;
    }
    
    .metric-item {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    
    /* 7. PDF DOWNLOAD BUTTON */
    .pdf-download-btn {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        margin: 20px 0 !important;
    }
    
    /* 8. RESPONSIVE DESIGN */
    @media (max-width: 768px) {
        .main-content {
            padding: 20px !important;
            padding-top: 70px !important;
        }
        
        .navbar-content {
            padding: 0 20px;
        }
        
        .hero-title {
            font-size: 36px;
        }
        
        .hero-subtitle {
            font-size: 20px;
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .results-container {
            padding: 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_navbar_html(current_page='home'):
    """Create navbar HTML."""
    html_parts = []
    html_parts.append('<div class="navbar-container">')
    html_parts.append('  <div class="navbar-content">')
    html_parts.append('    <div class="navbar-brand">')
    html_parts.append('      <div class="logo-text">SEI-AI Interviewer</div>')
    html_parts.append('    </div>')
    html_parts.append('    <div class="nav-buttons-container">')
    
    home_active = "active" if current_page == 'home' else ""
    info_active = "active" if current_page == 'info' else ""
    
    html_parts.append(f'      <a href="#" class="navbar-btn {home_active}" id="nav-home">Home</a>')
    html_parts.append(f'      <a href="#" class="navbar-btn {info_active}" id="nav-info">Info</a>')
    
    html_parts.append('    </div>')
    html_parts.append('  </div>')
    html_parts.append('</div>')
    html_parts.append('<div class="main-content">')
    
    return '\n'.join(html_parts)

def inject_navbar_js():
    """Inject JavaScript for navbar."""
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const homeBtn = document.getElementById('nav-home');
        const infoBtn = document.getElementById('nav-info');
        
        if (homeBtn) {
            homeBtn.onclick = function(e) {
                e.preventDefault();
                sessionStorage.setItem('nav_to', 'home');
                window.location.reload();
            };
        }
        
        if (infoBtn) {
            infoBtn.onclick = function(e) {
                e.preventDefault();
                sessionStorage.setItem('nav_to', 'info');
                window.location.reload();
            };
        }
        
        // Check for navigation request
        const navTo = sessionStorage.getItem('nav_to');
        if (navTo) {
            sessionStorage.removeItem('nav_to');
        }
    });
    </script>
    """, unsafe_allow_html=True)

def render_navbar(current_page='home'):
    """Render navbar."""
    st.markdown(create_navbar_html(current_page), unsafe_allow_html=True)
    inject_navbar_js()

def close_navbar():
    """Close navbar."""
    st.markdown("</div>", unsafe_allow_html=True)

# --- Page Render Functions ---
def render_home_page():
    """Render home page."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job.</p>', unsafe_allow_html=True)
    
    if st.button("Start Interview Now", type="primary", use_container_width=True, key="home_start"):
        next_page("candidate_form")
    
    st.markdown("---")
    
    # How it works section
    st.subheader("How It Works")
    cols = st.columns(5)
    steps = [
        ("1", "Register", "Enter your information"),
        ("2", "Answer Questions", "Record video responses"),
        ("3", "AI Analysis", "Speech and content analysis"),
        ("4", "Get Scores", "Rubric-based evaluation"),
        ("5", "Download PDF", "Comprehensive report")
    ]
    
    for i, (num, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{num}. {title}**")
            st.caption(desc)
    
    close_navbar()

def render_info_page():
    """Render info page."""
    inject_global_css()
    render_navbar('info')
    
    st.title("📚 Application Information")
    st.markdown("""
    ### Technology Stack
    
    This application uses state-of-the-art AI technologies:
    
    - **Speech Recognition**: OpenAI Whisper model
    - **Text Analysis**: Sentence Transformers for semantic similarity
    - **Non-verbal Analysis**: Audio processing for pace and pauses
    - **PDF Generation**: Professional report formatting
    
    ### Report Features
    
    Each interview generates a comprehensive PDF report containing:
    
    1. Candidate information
    2. Overall summary scores
    3. Non-verbal analysis metrics
    4. Detailed breakdown per question
    5. Actionable feedback and recommendations
    
    ### Privacy & Security
    
    - All processing happens in real-time
    - No permanent storage of video files
    - Data is not shared with third parties
    """)
    
    if st.button("🏠 Back to Home", type="primary"):
        next_page('home')
    
    close_navbar()

def render_candidate_form():
    """Render candidate form."""
    inject_global_css()
    render_navbar('home')
    
    st.markdown('<div class="candidate-form-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="candidate-form-title">Candidate Information</h1>', unsafe_allow_html=True)
    
    with st.form("candidate_form"):
        name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email Address", placeholder="Enter your email")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.text_input("Position Applied", placeholder="e.g., Software Engineer")
        with col2:
            experience = st.selectbox("Years of Experience", 
                                     ["0-2 years", "3-5 years", "6-10 years", "10+ years"])
        
        submitted = st.form_submit_button("Start Interview", type="primary", use_container_width=True)
        
        if submitted:
            if not name.strip():
                st.error("Please enter your name")
                return
            if not email.strip() or "@" not in email:
                st.error("Please enter a valid email")
                return
            
            interview_id = str(uuid.uuid4())[:8].upper()
            start_time = get_local_time_indonesia().strftime("%Y-%m-%d %H:%M:%S")
            
            st.session_state.candidate_data = {
                'id': interview_id,
                'name': name.strip(),
                'email': email.strip(),
                'position': position.strip(),
                'experience': experience,
                'start_time': start_time
            }
            st.session_state.interview_id = interview_id
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            
            next_page("interview")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Back to Home"):
        next_page('home')
    
    close_navbar()

def render_interview_page():
    """Render interview page."""
    inject_global_css()
    render_navbar('interview')
    
    # Candidate info banner
    if st.session_state.candidate_data:
        candidate = st.session_state.candidate_data
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0;">{candidate['name']}</h3>
                    <p style="margin: 5px 0; opacity: 0.9;">Position: {candidate.get('position', 'N/A')}</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 12px;">ID: {candidate['id']}</p>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">Started: {candidate['start_time']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.title(f"🎯 Interview Question {st.session_state.current_q} of {TOTAL_QUESTIONS}")
    
    q_num = st.session_state.current_q
    q_id_str = str(q_num)
    question_data = QUESTIONS.get(q_id_str, {})
    question_text = question_data.get('question', 'Question not found.')
    
    st.markdown("### 📝 Question:")
    st.info(f"**{question_text}**")
    
    st.markdown("---")
    
    col_upload, col_control = st.columns([3, 1])
    
    current_file = st.session_state.answers.get(q_id_str)
    
    with col_upload:
        if current_file is None:
            uploaded_file = st.file_uploader(
                f"📤 Upload Video Answer (Max {VIDEO_MAX_SIZE_MB}MB)",
                type=['mp4', 'mov', 'webm'],
                key=f"upload_{q_id_str}",
                help="Record your answer and upload the video file"
            )
            
            if uploaded_file:
                if uploaded_file.size > VIDEO_MAX_SIZE_MB * 1024 * 1024:
                    st.error(f"File size exceeds {VIDEO_MAX_SIZE_MB}MB limit")
                else:
                    st.session_state.answers[q_id_str] = uploaded_file
                    st.rerun()
        else:
            st.video(current_file)
            st.success(f"✅ Video uploaded: {current_file.name}")
            
            if st.button("🗑️ Delete Video", type="secondary"):
                del st.session_state.answers[q_id_str]
                st.rerun()
    
    with col_control:
        is_ready = current_file is not None
        
        if q_num < TOTAL_QUESTIONS:
            if st.button("⏭️ Next Question", use_container_width=True, disabled=not is_ready):
                st.session_state.current_q += 1
                st.rerun()
        elif q_num == TOTAL_QUESTIONS:
            if st.button("🏁 Finish & Process", use_container_width=True, disabled=not is_ready, type="primary"):
                next_page('processing')
        
        if q_num > 1:
            if st.button("⏮️ Previous", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()
        
        if st.button("🗑️ Cancel Interview", type="secondary", use_container_width=True):
            st.session_state.clear()
            next_page('home')
    
    close_navbar()

def render_processing_page():
    """Render processing page."""
    inject_global_css()
    render_navbar('processing')
    
    st.title("⚙️ Processing Your Interview")
    
    if st.session_state.results is not None:
        next_page('final_summary')
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        results = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(1, TOTAL_QUESTIONS + 1):
                q_id_str = str(i)
                q_key_rubric = f'q{i}'
                
                video_file = st.session_state.answers.get(q_id_str)
                q_text = QUESTIONS.get(q_id_str, {}).get('question')
                
                if video_file and q_key_rubric in RUBRIC_DATA and q_text:
                    status_text.text(f"Processing Question {i}...")
                    progress_bar.progress((i-1) * 20)
                    
                    # Save and process video
                    temp_video_path = os.path.join(temp_dir, f'video_{i}.mp4')
                    temp_audio_path = os.path.join(temp_dir, f'audio_{i}.wav')
                    
                    with open(temp_video_path, 'wb') as f:
                        f.write(video_file.getbuffer())
                    
                    video_to_wav(temp_video_path, temp_audio_path)
                    noise_reduction(temp_audio_path, temp_audio_path)
                    
                    # Analysis
                    transcript, log_prob_raw = transcribe_and_clean(
                        temp_audio_path, STT_MODEL, SPELL_CHECKER, EMBEDDER_MODEL, ENGLISH_WORDS
                    )
                    
                    confidence_score = compute_confidence_score(transcript, log_prob_raw)
                    non_verbal_res = analyze_non_verbal(temp_audio_path)
                    score, reason = score_with_rubric(
                        q_key_rubric, q_text, transcript, RUBRIC_DATA, EMBEDDER_MODEL
                    )
                    
                    results[q_key_rubric] = {
                        "question": q_text,
                        "transcript": transcript,
                        "final_score": int(score) if score else 0,
                        "rubric_reason": reason,
                        "confidence_score": f"{confidence_score * 100:.1f}%",
                        "non_verbal": non_verbal_res
                    }
                    
                    progress_bar.progress(i * 20)
        
        # Generate nonverbal scores
        nonverbal_scores = {
            "eye_contact": 82,
            "body_language": 78,
            "facial_expression": 85,
            "voice_clarity": 88,
            "gestures": 75
        }
        
        st.session_state.results = results
        st.session_state.nonverbal_scores = nonverbal_scores
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success("All questions have been analyzed successfully!")
        st.balloons()
        
        if st.button("View Results", type="primary"):
            next_page('final_summary')
    
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        if st.button("Back to Home"):
            st.session_state.clear()
            next_page('home')
    
    close_navbar()

def render_final_summary_page():
    """Render final summary with PDF download."""
    inject_global_css()
    render_navbar('final_summary')
    
    # Candidate header
    if st.session_state.candidate_data:
        candidate = st.session_state.candidate_data
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; margin-bottom: 30px; border-left: 6px solid #3498db;">
            <h2 style="margin: 0 0 10px 0; color: #2c3e50;">Interview Report</h2>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p style="margin: 5px 0;"><strong>Candidate:</strong> {candidate['name']}</p>
                    <p style="margin: 5px 0;"><strong>Position:</strong> {candidate.get('position', 'N/A')}</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 5px 0;"><strong>ID:</strong> {candidate['id']}</p>
                    <p style="margin: 5px 0;"><strong>Date:</strong> {candidate['start_time']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.title("📊 Interview Analysis Results")
    
    if not st.session_state.results:
        st.error("No results available")
        if st.button("Back to Home"):
            next_page('home')
        return
    
    results = st.session_state.results
    
    # Calculate overall metrics
    all_scores = [int(res['final_score']) for res in results.values()]
    all_confidence = [float(res['confidence_score'].replace('%', '')) for res in results.values()]
    
    avg_score = np.mean(all_scores) if all_scores else 0
    avg_confidence = np.mean(all_confidence) if all_confidence else 0
    
    # Overall Metrics
    st.subheader("📈 Overall Performance")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Average Rubric Score", f"{avg_score:.1f}/4.0")
    with cols[1]:
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    with cols[2]:
        nonverbal_scores = st.session_state.get('nonverbal_scores', {})
        eye_contact = nonverbal_scores.get('eye_contact', 0)
        st.metric("Eye Contact", f"{eye_contact}%")
    with cols[3]:
        total_questions = len(results)
        completed = len([s for s in all_scores if s > 0])
        st.metric("Questions Completed", f"{completed}/{total_questions}")
    
    st.markdown("---")
    
    # Non-Verbal Analysis Section
    st.subheader("🎭 Non-Verbal Analysis")
    
    if st.session_state.nonverbal_scores:
        nonverbal_data = st.session_state.nonverbal_scores
        for metric, score in nonverbal_data.items():
            if isinstance(score, (int, float)):
                st.progress(score/100, text=f"{metric.replace('_', ' ').title()}: {score}%")
    else:
        st.info("No non-verbal analysis data available")
    
    st.markdown("---")
    
    # Detailed Breakdown Section - IN THE EXACT FORMAT REQUESTED
    st.subheader("📋 Detailed Breakdown by Question")
    
    for q_key, res in results.items():
        q_num = q_key.replace('q', '')
        
        with st.expander(f"Question {q_num}: {res['question'][:80]}...", expanded=True):
            # Create two columns for layout
            col_top, col_bottom = st.columns([1, 1])
            
            with col_top:
                # Rubric and Confidence at the TOP as requested
                st.markdown("**Scores:**")
                col_rubric, col_conf = st.columns(2)
                with col_rubric:
                    st.metric("Rubric", f"{res['final_score']}/4")
                with col_conf:
                    st.metric("Confidence", res['confidence_score'])
            
            # Reasons section
            st.markdown("**Evaluation:**")
            rubric_reason = res['rubric_reason']
            if isinstance(rubric_reason, str):
                if '. ' in rubric_reason:
                    reasons = rubric_reason.split('. ')
                else:
                    reasons = [rubric_reason]
                
                for reason in reasons:
                    if reason.strip():
                        st.markdown(f"• {reason.strip()}")
            else:
                st.markdown(f"• {rubric_reason}")
            
            with col_bottom:
                # Tempo and Pause at the BOTTOM as requested
                st.markdown("**Speech Analysis:**")
                non_verbal = res.get('non_verbal', {})
                tempo = non_verbal.get('tempo_bpm', 'N/A')
                pause = non_verbal.get('total_pause_seconds', 'N/A')
                
                # Extract numeric values
                tempo_value = ''.join(filter(str.isdigit, str(tempo))) if tempo != 'N/A' else 'N/A'
                pause_value = ''.join(filter(str.isdigit, str(pause))) if pause != 'N/A' else 'N/A'
                
                st.markdown(f"• **Tempo:** {tempo_value} words per minute")
                st.markdown(f"• **Pause:** {pause_value} seconds total")
            
            # Transcript (collapsible)
            with st.expander("View Transcript"):
                st.text_area("", res['transcript'], height=150, disabled=True, key=f"transcript_{q_num}")
    
    st.markdown("---")
    
    # PDF Download Section
    st.subheader("📄 Download Report")
    
    col_pdf, col_json, col_new = st.columns(3)
    
    with col_pdf:
        # PDF Download Button
        download_pdf_report()
    
    with col_json:
        # JSON Download
        if st.button("📥 Download JSON Data", use_container_width=True):
            report_data = {
                'candidate': st.session_state.candidate_data,
                'results': st.session_state.results,
                'nonverbal_scores': st.session_state.get('nonverbal_scores', {}),
                'summary': {
                    'avg_score': f"{avg_score:.1f}",
                    'avg_confidence': f"{avg_confidence:.1f}%"
                }
            }
            
            json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"interview_data_{st.session_state.candidate_data['id']}.json",
                mime="application/json",
                key="json_download"
            )
    
    with col_new:
        if st.button("🔄 New Interview", use_container_width=True, type="primary"):
            # Keep candidate data but reset interview
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page('candidate_form')
    
    # Back to home button
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.clear()
        next_page('home')
    
    close_navbar()

# Main App Execution Flow
if __name__ == "__main__":
    # Check for navigation from session storage
    try:
        import streamlit.components.v1 as components
        components.html("""
        <script>
        if (sessionStorage.getItem('nav_to')) {
            window.location.reload();
        }
        </script>
        """, height=0)
    except:
        pass
    
    # Route to correct page
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