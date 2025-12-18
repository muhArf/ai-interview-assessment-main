import streamlit as st
import pandas as pd
import json
import os
import tempfile
import sys
import numpy as np
import base64
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Report Generation Functions ---
def generate_evaluation_report():
    """Generate comprehensive evaluation report as HTML."""
    if not st.session_state.results:
        return None
    
    # Calculate overall metrics
    all_scores = []
    all_confidence = []
    all_tempo = []
    all_pause = []
    
    for res in st.session_state.results.values():
        all_scores.append(int(res['final_score']))
        all_confidence.append(float(res['confidence_score'].replace('%', '')))
        
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
    total_pause = np.sum(all_pause) if all_pause else 0
    
    # Determine overall performance level
    if avg_score >= 3.5:
        performance_level = "Excellent"
        performance_color = "#2ecc71"
    elif avg_score >= 2.5:
        performance_level = "Good"
        performance_color = "#f39c12"
    else:
        performance_level = "Needs Improvement"
        performance_color = "#e74c3c"
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SEI-AI Evaluation Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 32px;
            }}
            .header .subtitle {{
                font-size: 18px;
                opacity: 0.9;
                margin-top: 10px;
            }}
            .summary-card {{
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 5px solid {performance_color};
            }}
            .performance-badge {{
                display: inline-block;
                background-color: {performance_color};
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 18px;
                margin: 10px 0;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }}
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .question-section {{
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                border-left: 4px solid #3498db;
            }}
            .question-header {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                border-left: 4px solid #3498db;
            }}
            .score-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 15px;
                font-weight: bold;
                margin-right: 10px;
                font-size: 14px;
            }}
            .score-excellent {{ background-color: #2ecc71; color: white; }}
            .score-good {{ background-color: #f39c12; color: white; }}
            .score-fair {{ background-color: #e74c3c; color: white; }}
            .analysis-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .analysis-box {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #e9ecef;
            }}
            .transcript-box {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                border: 1px solid #e9ecef;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .recommendations {{
                background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
                padding: 25px;
                border-radius: 10px;
                margin: 30px 0;
                border-left: 5px solid #f39c12;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 14px;
            }}
            @media print {{
                body {{
                    background-color: white;
                }}
                .metric-card, .question-section {{
                    break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 SEI-AI Interview Evaluation Report</h1>
            <div class="subtitle">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
        </div>
        
        <div class="summary-card">
            <h2>Overall Performance Summary</h2>
            <div class="performance-badge">{performance_level}</div>
            <p>Based on the analysis of all {TOTAL_QUESTIONS} interview questions, here is your comprehensive performance evaluation.</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" style="color: #2ecc71;">{avg_score:.2f}/4.0</div>
                    <div class="metric-label">Average Content Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #3498db;">{avg_confidence:.2f}%</div>
                    <div class="metric-label">Transcript Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #f39c12;">{avg_tempo:.1f} BPM</div>
                    <div class="metric-label">Average Speaking Tempo</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #e74c3c;">{total_pause:.1f} sec</div>
                    <div class="metric-label">Total Pause Time</div>
                </div>
            </div>
        </div>
    """
    
    # Add detailed question analysis
    html_content += """
        <h2>📋 Detailed Question Analysis</h2>
    """
    
    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        score = res['final_score']
        
        # Determine score badge class
        if score >= 3.5:
            score_class = "score-excellent"
            score_label = "Excellent"
        elif score >= 2.5:
            score_class = "score-good"
            score_label = "Good"
        else:
            score_class = "score-fair"
            score_label = "Needs Work"
        
        html_content += f"""
        <div class="question-section">
            <div class="question-header">
                <h3>Question {q_num} <span class="score-badge {score_class}">{score}/4 - {score_label}</span></h3>
                <p><strong>Question:</strong> {res['question']}</p>
            </div>
            
            <div class="analysis-grid">
                <div class="analysis-box">
                    <h4>🎯 Content Analysis</h4>
                    <p><strong>Score:</strong> {score}/4</p>
                    <p><strong>Evaluation:</strong> {res['rubric_reason']}</p>
                </div>
                
                <div class="analysis-box">
                    <h4>🎤 Non-Verbal Analysis</h4>
                    <p><strong>Speaking Tempo:</strong> {res['non_verbal'].get('tempo_bpm', 'N/A')}</p>
                    <p><strong>Pause Time:</strong> {res['non_verbal'].get('total_pause_seconds', 'N/A')}</p>
                    <p><strong>Summary:</strong> {res['non_verbal'].get('qualitative_summary', 'N/A')}</p>
                </div>
                
                <div class="analysis-box">
                    <h4>📝 Technical Analysis</h4>
                    <p><strong>Transcript Confidence:</strong> {res['confidence_score']}</p>
                    <p><strong>Model Assessment:</strong> AI analysis based on semantic similarity and rubric matching</p>
                </div>
            </div>
            
            <div class="transcript-box">
                <strong>Transcript:</strong><br>
                {res['transcript']}
            </div>
        </div>
        """
    
    # Add recommendations section
    recommendations = []
    
    if avg_score < 3.0:
        recommendations.append("""
        <li><strong>Improve Answer Structure:</strong> Use the STAR method (Situation, Task, Action, Result) to structure your responses more effectively.</li>
        <li><strong>Increase Specificity:</strong> Include concrete examples with measurable outcomes and specific details.</li>
        <li><strong>Align with Rubrics:</strong> Review the expected criteria for each question type and tailor your answers accordingly.</li>
        """)
    
    if avg_tempo > 150:
        recommendations.append("""
        <li><strong>Slow Down Your Pace:</strong> Practice speaking at 120-140 words per minute for optimal comprehension.</li>
        <li><strong>Use Strategic Pauses:</strong> Incorporate brief pauses after key points to allow absorption of information.</li>
        """)
    elif avg_tempo < 120:
        recommendations.append("""
        <li><strong>Increase Speaking Pace:</strong> Aim for a more energetic delivery to maintain listener engagement.</li>
        <li><strong>Reduce Unnecessary Pauses:</strong> Minimize filler words and extended hesitations.</li>
        """)
    
    if total_pause > 120:
        recommendations.append("""
        <li><strong>Reduce Total Pause Time:</strong> Practice speaking more continuously while maintaining clarity.</li>
        <li><strong>Plan Your Responses:</strong> Structure your thoughts before speaking to reduce hesitation time.</li>
        """)
    
    if avg_confidence < 90:
        recommendations.append("""
        <li><strong>Improve Audio Clarity:</strong> Record in a quiet environment with minimal background noise.</li>
        <li><strong>Enunciate Clearly:</strong> Focus on pronouncing words fully and distinctly.</li>
        <li><strong>Consider Equipment:</strong> Use a high-quality microphone for better audio capture.</li>
        """)
    
    # Default recommendations if none generated
    if not recommendations:
        recommendations.append("""
        <li><strong>Maintain Current Performance:</strong> Continue practicing regularly to maintain your strong interview skills.</li>
        <li><strong>Expand Your Examples:</strong> Develop additional diverse examples for different question types.</li>
        <li><strong>Practice Time Management:</strong> Work on delivering complete answers within typical interview timeframes.</li>
        """)
    
    html_content += f"""
        <div class="recommendations">
            <h2>💡 Actionable Recommendations</h2>
            <p>Based on your performance analysis, here are specific areas for improvement:</p>
            <ul>
                {''.join(recommendations)}
            </ul>
            
            <h3>Practice Strategy:</h3>
            <p><strong>Immediate Focus (Next 7 days):</strong> Work on the highest-priority items identified above.</p>
            <p><strong>Medium-Term Goals (Next 30 days):</strong> Record practice sessions weekly and compare progress.</p>
            <p><strong>Long-Term Development (Next 90 days):</strong> Master multiple interview formats and question types.</p>
        </div>
        
        <div class="footer">
            <p><strong>SEI-AI Interview Evaluation System</strong></p>
            <p>This report was generated automatically by AI analysis. The evaluation is based on machine learning models 
            analyzing speech patterns, content relevance, and communication effectiveness.</p>
            <p>© {datetime.now().year} SEI-AI. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def create_download_button():
    """Create a download button for the evaluation report."""
    html_content = generate_evaluation_report()
    
    if html_content:
        # Create a download link for the HTML report
        b64_html = base64.b64encode(html_content.encode()).decode()
        
        # Create a PDF-like document in a more portable format
        # We'll create a styled HTML file that can be printed as PDF
        st.markdown(
            f"""
            <style>
            .download-btn {{
                display: inline-block;
                background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                color: white;
                padding: 14px 28px;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
                text-align: center;
                margin: 10px 0;
            }}
            .download-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
            }}
            </style>
            
            <a href="data:text/html;base64,{b64_html}" download="SEI-AI_Evaluation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html" 
               class="download-btn">
               📥 Download Full Evaluation Report (HTML)
            </a>
            """,
            unsafe_allow_html=True
        )
        
        # Also create a simplified text version
        text_report = generate_text_report()
        b64_text = base64.b64encode(text_report.encode()).decode()
        
        st.markdown(
            f"""
            <a href="data:text/plain;base64,{b64_text}" download="SEI-AI_Summary_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt" 
               class="download-btn" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); margin-left: 10px;">
               📄 Download Summary Report (TXT)
            </a>
            """,
            unsafe_allow_html=True
        )

def generate_text_report():
    """Generate a simplified text version of the report."""
    if not st.session_state.results:
        return "No results available."
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SEI-AI INTERVIEW EVALUATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Overall summary
    all_scores = [int(res['final_score']) for res in st.session_state.results.values()]
    avg_score = np.mean(all_scores) if all_scores else 0
    
    report_lines.append("OVERALL PERFORMANCE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Average Score: {avg_score:.2f}/4.0")
    report_lines.append(f"Number of Questions: {len(st.session_state.results)}")
    report_lines.append("")
    
    # Detailed breakdown
    report_lines.append("DETAILED QUESTION ANALYSIS")
    report_lines.append("=" * 70)
    
    for q_key, res in st.session_state.results.items():
        q_num = q_key.replace('q', '')
        report_lines.append(f"\nQuestion {q_num}:")
        report_lines.append(f"Question: {res['question']}")
        report_lines.append(f"Score: {res['final_score']}/4")
        report_lines.append(f"Transcript Confidence: {res['confidence_score']}")
        report_lines.append(f"Non-Verbal: {res['non_verbal'].get('qualitative_summary', 'N/A')}")
        report_lines.append(f"Evaluation: {res['rubric_reason']}")
        report_lines.append(f"Transcript: {res['transcript'][:200]}...")
        report_lines.append("-" * 50)
    
    # Recommendations
    report_lines.append("\nRECOMMENDATIONS")
    report_lines.append("=" * 70)
    
    if avg_score < 3.0:
        report_lines.append("1. Improve answer structure using STAR method")
        report_lines.append("2. Include more specific examples with measurable outcomes")
        report_lines.append("3. Align answers more closely with rubric criteria")
    
    report_lines.append("\nGENERAL ADVICE:")
    report_lines.append("- Practice regularly with different question types")
    report_lines.append("- Record and review your practice sessions")
    report_lines.append("- Focus on both content and delivery aspects")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("End of Report")
    report_lines.append("© SEI-AI Interview Evaluation System")
    
    return "\n".join(report_lines)

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
    
    /* 5. HOW IT WORKS SECTION - STEP CARDS */
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
    justify-content: space-between; /* Menggunakan space-between untuk distribusi merata */
    width: 100%;
    max-width: 1400px;
    gap: 20px; /* Gap antar card */
}

.step-card {
    background: white;
    border-radius: 20px;
    padding: 60px 25px 35px 25px; /* Atas lebih besar untuk number, bawah konsisten */
    text-align: center;
    width: 250px; /* Lebar tetap */
    height: 320px; /* Tinggi tetap - SEMUA CARD SAMA TINGGI */
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    position: relative;
    transition: all 0.4s ease;
    border: 1px solid #f0f0f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start; /* Konten mulai dari atas */
    flex: 0 0 auto; /* Tidak fleksibel, ukuran tetap */
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
    margin: 0 0 15px 0; /* Margin atas dihapus karena sudah ada padding atas */
    color: #000000;
    line-height: 1.4;
    text-align: center;
    width: 100%;
    min-height: 60px; /* Tinggi minimum untuk judul */
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
    flex-grow: 1; /* Deskripsi mengambil ruang yang tersisa */
    display: flex;
    align-items: center; /* Vertikal center untuk teks */
    justify-content: center; /* Horizontal center untuk teks */
    padding: 0 5px; /* Sedikit padding samping */
}

/* Responsive design untuk step-card */
@media (max-width: 1400px) {
    .step-card-wrapper {
        gap: 15px;
    }
    
    .step-card {
        width: 200px;
        height: 300px;
        padding: 55px 20px 30px 20px;
    }
}

@media (max-width: 1200px) {
    .step-card-wrapper {
        flex-wrap: wrap; /* Allow wrapping jika tidak cukup space */
        justify-content: center;
        gap: 25px;
        max-width: 1000px;
    }
    
    .step-card {
        width: 180px;
        height: 290px;
        padding: 50px 15px 25px 15px;
    }
    
    .step-title {
        font-size: 18px;
        min-height: 55px;
    }
    
    .step-description {
        font-size: 14px;
    }
}

@media (max-width: 992px) {
    .step-card-wrapper {
        gap: 20px;
    }
    
    .step-card {
        width: 170px;
        height: 280px;
        padding: 45px 12px 20px 12px;
    }
    
    .step-title {
        font-size: 17px;
        min-height: 50px;
        margin: 0 0 12px 0;
    }
    
    .step-description {
        font-size: 13px;
        line-height: 1.5;
    }
}

@media (max-width: 768px) {
    .step-card-container {
        padding: 0 15px;
    }
    
    .step-card-wrapper {
        gap: 20px;
        max-width: 600px;
    }
    
    .step-card {
        width: 160px;
        height: 270px;
        padding: 40px 10px 18px 10px;
    }
    
    .step-number {
        width: 45px;
        height: 45px;
        font-size: 20px;
        top: -22px;
    }
    
    .step-title {
        font-size: 16px;
        min-height: 45px;
        margin: 0 0 10px 0;
    }
    
    .step-description {
        font-size: 13px;
        line-height: 1.4;
    }
}

@media (max-width: 640px) {
    .step-card-wrapper {
        flex-direction: column;
        align-items: center;
        gap: 50px; /* Gap lebih besar untuk mobile */
        max-width: 400px;
    }
    
    .step-card {
        width: 100%;
        max-width: 300px;
        height: auto; /* Biarkan tinggi otomatis di mobile */
        min-height: 280px;
        padding: 55px 25px 30px 25px;
    }
    
    .step-number {
        width: 50px;
        height: 50px;
        font-size: 22px;
        top: -25px;
    }
    
    .step-title {
        font-size: 20px;
        min-height: auto;
        margin: 0 0 15px 0;
    }
    
    .step-description {
        font-size: 15px;
        line-height: 1.6;
    }
}

@media (max-width: 480px) {
    .step-card {
        padding: 50px 20px 25px 20px;
        max-width: 280px;
        min-height: 260px;
    }
}
    
    /* 6. FEATURES SECTION */
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
    
    /* 7. FOOTER */
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
    
    /* 8. METRIC CARDS FOR RESULTS - PERBAIKAN DI SINI */
    /* Container untuk metric cards */
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
        flex-wrap: nowrap; /* Tidak boleh wrap di desktop */
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        flex: 1; /* Mengambil ruang yang sama */
        min-width: 0; /* Agar tidak meledak */
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
    
    /* 9. INTERVIEW PAGE STYLING */
    .question-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 40px;
        border-left: 6px solid #667eea;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    /* 10. DOWNLOAD BUTTON STYLING */
    .download-container {
        background: linear-gradient(135deg, #f8fff9 0%, #f0fff4 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 30px 0;
        border: 2px solid #2ecc71;
        box-shadow: 0 5px 20px rgba(46, 204, 113, 0.15);
    }
    
    .download-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .download-icon {
        font-size: 32px;
    }
    
    .download-title {
        font-size: 24px;
        font-weight: 700;
        color: #27ae60;
        margin: 0;
    }
    
    .download-description {
        color: #666;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    
    .download-buttons {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
    }
    
    /* 11. RESPONSIVE DESIGN */
    @media (max-width: 1200px) {
        .main-content {
            padding-left: 30px !important;
            padding-right: 30px !important;
        }
        
        .metric-wrapper {
            flex-wrap: wrap; /* Boleh wrap di tablet */
        }
        
        .metric-card {
            flex: 0 0 calc(50% - 10px); /* 2 cards per row */
            min-width: 250px;
        }
        
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .download-buttons {
            flex-direction: column;
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
        
        .hero-section {
            padding: 40px 0 60px 0;
        }
        
        .section-title {
            font-size: 32px;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
        }
        
        .metric-wrapper {
            flex-direction: column; /* Stack vertical di mobile */
        }
        
        .metric-card {
            width: 100%;
            flex: 1 0 auto;
        }
        
        .custom-footer {
            flex-direction: column;
            gap: 20px;
            text-align: center;
            padding: 30px 20px;
        }
        
        .nav-buttons-container {
            gap: 10px;
        }
        
        .navbar-btn {
            min-width: 80px;
            padding: 6px 16px;
            font-size: 13px;
        }
        
        .download-container {
            padding: 20px;
        }
        
        .download-title {
            font-size: 20px;
        }
    }
    
    @media (max-width: 480px) {
        .main-content {
            padding-left: 15px !important;
            padding-right: 15px !important;
        }
        
        .logo-text {
            font-size: 20px;
        }
        
        .hero-title {
            font-size: 36px;
        }
        
        .hero-subtitle {
            font-size: 24px;
        }
        
        .nav-buttons-container {
            gap: 5px;
        }
        
        .navbar-btn {
            min-width: 70px;
            padding: 5px 12px;
            font-size: 12px;
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

def render_home_page():
    """Render the fixed landing page."""
    inject_global_css()
    
    # Render navbar dengan semua elemen dalam HTML
    render_navbar('home')
    
    # HERO SECTION

    
    st.markdown('<h1 class="hero-title">Welcome to SEI-AI Interviewer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Hone your interview skills with AI-powered feedback and prepare for your dream job with comprehensive evaluation and actionable insights.</p>', unsafe_allow_html=True)
    
    if st.button("Start Interview Now", key="hero_start", type="primary"):
        st.session_state.answers = {}
        st.session_state.results = None
        st.session_state.current_q = 1
        next_page("interview")
    
    st.markdown('</section>', unsafe_allow_html=True)
    
    # HOW IT WORKS SECTION
    st.markdown('<div class="text-center" style="margin-bottom: 40px;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">How To Use</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Container untuk step cards dengan flex layout yang lebih baik
    st.markdown('<div class="step-card-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-card-wrapper">', unsafe_allow_html=True)
    
    steps = [
        ("1", "Upload Answer Video", "Upload your video answer for each interview question provided by the system."),
        ("2", "AI Processing", "The AI processes video into transcript and analyzes non-verbal communication aspects."),
        ("3", "Semantic Scoring", "Your answer is compared to ideal rubric criteria for content relevance scoring."),
        ("4", "Get Instant Feedback", "Receive final score, detailed rationale, and communication analysis immediately."),
        ("5", "Improve Your Skills", "Use personalized recommendations to practice and enhance your interview performance.")
    ]
    
    # Menggunakan st.columns dengan 5 kolom untuk desktop
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
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close step-card-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # Close step-card-container
    
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
    # Gunakan render_navbar biasa untuk halaman info
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
    """Render the interview page."""
    inject_global_css()
    # Gunakan render_navbar biasa untuk halaman interview
    render_navbar('interview')
    
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
    """Render the processing page."""
    inject_global_css()
    # Gunakan render_navbar biasa untuk halaman processing
    render_navbar('processing')
    
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
    """Render the final results page with enhanced report download feature."""
    inject_global_css()
    # Gunakan render_navbar biasa untuk halaman final summary
    render_navbar('final_summary')
    
    st.title("🏆 Final Evaluation Report")
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
    
    # PERBAIKAN DI SINI: Menggunakan st.columns untuk metric cards horizontal
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-wrapper">', unsafe_allow_html=True)
    
    # Metric cards dalam satu baris horizontal
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
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close metric-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # Close metric-container
    
    # --- DOWNLOAD REPORT SECTION ---
    st.markdown("---")
    st.subheader("📥 Download Detailed Report")
    
    # Create a styled download container
    st.markdown('<div class="download-container">', unsafe_allow_html=True)
    st.markdown("""
        <div class="download-header">
            <div class="download-icon">📊</div>
            <h3 class="download-title">Comprehensive Evaluation Report</h3>
        </div>
        <p class="download-description">
            Download a detailed report containing complete analysis of each question, including:
            • Full question-by-question breakdown<br>
            • Transcript analysis with confidence scores<br>
            • Non-verbal communication assessment<br>
            • Rubric-based scoring rationale<br>
            • Personalized improvement recommendations
        </p>
    """, unsafe_allow_html=True)
    
    # Generate and display download buttons
    create_download_button()
    
    st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
            <strong>📋 Report Features:</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>Professional HTML format with full styling</li>
                <li>Printable design for physical records</li>
                <li>Includes all transcripts and evaluations</li>
                <li>Actionable recommendations for improvement</li>
                <li>Timestamped for progress tracking</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close download-container
    
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
    with st.expander("📋 View Detailed Breakdown by Question", expanded=True):
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
            st.session_state.clear()
            next_page('home')
    
    with col_btn2:
        # Preview button for the report
        if st.button("👁️ Preview Report", use_container_width=True):
            # Generate and display a preview of the report
            html_content = generate_evaluation_report()
            if html_content:
                # Display first 2000 characters as preview
                preview_text = html_content[:2000] + "..." if len(html_content) > 2000 else html_content
                with st.expander("Report Preview", expanded=True):
                    st.markdown("**Note:** This is a preview. Download the full report for complete details.")
                    st.components.v1.html(preview_text, height=300, scrolling=True)
    
    with col_btn3:
        if st.button("🏠 Back to Home", use_container_width=True):
            next_page('home')
    
    close_navbar()

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