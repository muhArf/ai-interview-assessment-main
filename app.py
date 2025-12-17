# Bagian yang dimodifikasi dari app.py
# Ganti fungsi inject_global_css() dengan versi ini:

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
        background: #f8f9fa;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    
    /* 2. FIXED NAVBAR CONTAINER */
    .navbar-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 70px;
        display: flex;
        align-items: center;
    }
    
    .navbar-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    
    /* Brand/Logo */
    .navbar-brand {
        display: flex;
        align-items: center;
        height: 50px;
    }
    
    .navbar-logo {
        height: 40px;
        width: auto;
        object-fit: contain;
    }
    
    /* Navigation buttons container */
    .nav-buttons-container {
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
        background: transparent !important;
        color: #667eea !important;
        padding: 8px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        height: 40px !important;
        min-width: 100px;
    }
    
    .stButton > button:hover {
        background: #667eea !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        border-color: #667eea !important;
    }
    
    /* 3. MAIN CONTENT PADDING */
    .main-content {
        padding-top: 70px;
    }
    
    /* 4. LANDING PAGE HERO SECTION */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 100px 40px;
        text-align: center;
        position: relative;
        color: white;
        margin-bottom: 80px;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 24px;
        line-height: 1.2;
        letter-spacing: -1px;
        text-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
    }
    
    .hero-subtitle {
        font-size: 20px;
        max-width: 700px;
        margin: 0 auto 48px auto;
        line-height: 1.6;
        font-weight: 400;
        opacity: 0.95;
    }
    
    .hero-buttons {
        display: flex;
        gap: 20px;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .primary-btn {
        background: white !important;
        color: #667eea !important;
        border-radius: 30px !important;
        padding: 16px 48px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .primary-btn:hover {
        background: #f8f9fa !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .secondary-btn {
        background: transparent !important;
        color: white !important;
        border: 2px solid white !important;
        border-radius: 30px !important;
        padding: 16px 48px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    
    .secondary-btn:hover {
        background: white !important;
        color: #667eea !important;
        transform: translateY(-3px);
    }
    
    /* 5. STATS SECTION */
    .stats-section {
        max-width: 1200px;
        margin: -60px auto 80px auto;
        padding: 0 40px;
        position: relative;
        z-index: 2;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
    }
    
    .stat-item {
        text-align: center;
        padding: 20px;
    }
    
    .stat-number {
        font-size: 42px;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 8px;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
        font-weight: 500;
    }
    
    /* 6. HOW IT WORKS SECTION */
    .section-wrapper {
        max-width: 1200px;
        margin: 0 auto;
        padding: 60px 40px;
    }
    
    .section-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 16px;
        color: #1a1a1a;
        letter-spacing: -0.5px;
    }
    
    .section-subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 60px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .steps-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 25px;
        margin-bottom: 40px;
    }
    
    .step-card {
        background: white;
        border-radius: 20px;
        padding: 35px 25px;
        text-align: center;
        min-height: 280px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        position: relative;
        transition: all 0.4s ease;
        border: 2px solid transparent;
    }
    
    .step-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .step-icon {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 700;
        margin: 0 auto 20px auto;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.25);
    }
    
    .step-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #1a1a1a;
    }
    
    .step-description {
        color: #666;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* 7. FEATURES SECTION */
    .features-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 80px 0;
        margin-top: 60px;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 40px 30px;
        text-align: left;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 20px;
        display: block;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #1a1a1a;
    }
    
    .feature-desc {
        color: #666;
        font-size: 15px;
        line-height: 1.6;
    }
    
    /* 8. CTA SECTION */
    .cta-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 80px 40px;
        text-align: center;
        margin-top: 80px;
    }
    
    .cta-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 20px;
    }
    
    .cta-subtitle {
        font-size: 18px;
        margin-bottom: 40px;
        opacity: 0.95;
    }
    
    /* 9. FOOTER */
    .custom-footer {
        background: #1a1a1a;
        color: white;
        padding: 50px 40px 30px 40px;
    }
    
    .footer-content {
        max-width: 1200px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 40px;
        margin-bottom: 40px;
    }
    
    .footer-brand {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer-desc {
        font-size: 14px;
        opacity: 0.7;
        line-height: 1.6;
    }
    
    .footer-section-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 16px;
    }
    
    .footer-links {
        list-style: none;
        padding: 0;
    }
    
    .footer-links li {
        margin-bottom: 12px;
    }
    
    .footer-links a {
        color: rgba(255, 255, 255, 0.7);
        text-decoration: none;
        font-size: 14px;
        transition: color 0.3s;
    }
    
    .footer-links a:hover {
        color: white;
    }
    
    .footer-bottom {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 30px;
        text-align: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .footer-copyright {
        font-size: 14px;
        opacity: 0.6;
    }
    
    /* 10. METRIC CARDS FOR RESULTS */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 30px 0 50px 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666666;
        font-weight: 500;
    }
    
    .score-color { color: #2ecc71; }
    .accuracy-color { color: #3498db; }
    .tempo-color { color: #f39c12; }
    .pause-color { color: #e74c3c; }
    
    /* 11. INTERVIEW PAGE STYLING */
    .question-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }
    
    /* 12. RESPONSIVE DESIGN */
    @media (max-width: 1200px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .footer-content {
            grid-template-columns: 1fr 1fr;
        }
    }
    
    @media (max-width: 768px) {
        .navbar-content {
            padding: 0 20px;
        }
        
        .hero-section {
            padding: 60px 20px;
        }
        
        .hero-title {
            font-size: 36px;
        }
        
        .hero-subtitle {
            font-size: 16px;
        }
        
        .section-title {
            font-size: 32px;
        }
        
        .steps-container {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
            padding: 30px 20px;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .footer-content {
            grid-template-columns: 1fr;
        }
        
        .hero-buttons {
            flex-direction: column;
        }
        
        .primary-btn, .secondary-btn {
            width: 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# Ganti fungsi render_home_page() dengan versi ini:

def render_home_page():
    """Render the improved landing page."""
    inject_global_css()
    render_navbar()
    
    # HERO SECTION
    st.markdown("""
    <section class="hero-section">
        <div class="hero-content">
            <div class="hero-badge">🎯 AI-Powered Interview Practice</div>
            <h1 class="hero-title">Master Your Interview Skills with SEI-AI</h1>
            <p class="hero-subtitle">Get comprehensive AI-powered feedback on your interview performance. Improve your content, delivery, and confidence with personalized insights and actionable recommendations.</p>
        </div>
    </section>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("▶️ Start Your Practice Now", key="hero_start", use_container_width=True):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page("interview")
    
    # STATS SECTION
    st.markdown("""
    <div class="stats-section">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">95%</div>
                <div class="stat-label">Accuracy Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">5+</div>
                <div class="stat-label">Question Types</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">3</div>
                <div class="stat-label">Analysis Metrics</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">< 5min</div>
                <div class="stat-label">Processing Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # HOW IT WORKS SECTION
    st.markdown("""
    <div class="section-wrapper">
        <h2 class="section-title">How It Works</h2>
        <p class="section-subtitle">Follow these simple steps to get comprehensive feedback on your interview performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-wrapper"><div class="steps-container">', unsafe_allow_html=True)
    
    steps = [
        ("1", "📤", "Upload Video", "Record and upload your video answer for each interview question"),
        ("2", "🤖", "AI Processing", "Advanced AI converts speech to text and analyzes your delivery"),
        ("3", "📊", "Semantic Scoring", "Content is evaluated against industry-standard rubrics"),
        ("4", "💡", "Get Feedback", "Receive detailed scores, insights, and improvement tips"),
        ("5", "🚀", "Practice & Improve", "Apply recommendations and track your progress")
    ]
    
    cols = st.columns(5)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{num}</div>
                <h3 class="step-title">{title}</h3>
                <p class="step-description">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # KEY FEATURES SECTION
    st.markdown("""
    <section class="features-section">
        <div class="section-wrapper">
            <h2 class="section-title">Powerful Features</h2>
            <p class="section-subtitle">Everything you need to ace your next interview</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="features-grid">', unsafe_allow_html=True)
    
    features = [
        ("🎤", "Advanced Speech Recognition", "Industry-leading Whisper AI ensures accurate transcription of your responses"),
        ("📊", "Multi-Dimensional Analysis", "Evaluate content quality, delivery speed, and vocal characteristics simultaneously"),
        ("⚡", "Instant Results", "Get comprehensive feedback within minutes, not days"),
        ("🎯", "Rubric-Based Scoring", "Objective evaluation using proven interview assessment frameworks"),
        ("📈", "Progress Tracking", "Monitor your improvement over multiple practice sessions"),
        ("🔒", "Privacy First", "Your videos are processed securely and never permanently stored")
    ]
    
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <h3 class="feature-title">{title}</h3>
                <p class="feature-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div></section>', unsafe_allow_html=True)
    
    # CTA SECTION
    st.markdown("""
    <section class="cta-section">
        <h2 class="cta-title">Ready to Elevate Your Interview Game?</h2>
        <p class="cta-subtitle">Join thousands of professionals who have improved their interview skills with SEI-AI</p>
    </section>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🎯 Get Started Free", key="cta_start", use_container_width=True):
            st.session_state.answers = {}
            st.session_state.results = None
            st.session_state.current_q = 1
            next_page("interview")
    
    # FOOTER
    st.markdown("""
    <footer class="custom-footer">
        <div class="footer-content">
            <div>
                <div class="footer-brand">SEI-AI</div>
                <p class="footer-desc">Empowering professionals with AI-driven interview preparation and feedback. Practice with confidence.</p>
            </div>
            <div>
                <h4 class="footer-section-title">Product</h4>
                <ul class="footer-links">
                    <li><a href="#">Features</a></li>
                    <li><a href="#">How it Works</a></li>
                    <li><a href="#">Pricing</a></li>
                </ul>
            </div>
            <div>
                <h4 class="footer-section-title">Resources</h4>
                <ul class="footer-links">
                    <li><a href="#">Blog</a></li>
                    <li><a href="#">Help Center</a></li>
                    <li><a href="#">Contact</a></li>
                </ul>
            </div>
            <div>
                <h4 class="footer-section-title">Legal</h4>
                <ul class="footer-links">
                    <li><a href="#">Privacy Policy</a></li>
                    <li><a href="#">Terms of Service</a></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <p class="footer-copyright">© 2024 SEI-AI Interviewer. All rights reserved.</p>
        </div>
    </footer>
    """, unsafe_allow_html=True)
    
    close_navbar()