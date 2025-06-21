# cyber_japan_theme.py
"""
Cyber Japan Theme для SMPP Anomaly Detection Dashboard
Спрощена неонова кіберпанк тема
"""

def get_cyber_japan_css():
    """Повертає CSS для Cyber Japan теми"""
    return """
<style>
    /* === CYBER JAPAN THEME (SIMPLIFIED) === */
    
    /* Основний фон */
    .stApp {
        background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
        color: #F8F9FA;
    }
    
    /* Картки метрик */
    [data-testid="metric-container"] {
        background: rgba(26, 26, 26, 0.9);
        border: 2px solid #FF1744;
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: #FF6B35;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.5);
        transform: translateY(-2px);
    }
    
    /* Заголовки */
    h1, h2, h3 {
        font-family: 'Courier New', monospace !important;
        background: linear-gradient(45deg, #FF1744, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(255, 23, 68, 0.5);
        letter-spacing: 1px;
        font-weight: bold !important;
    }
    
    /* Алерти */
    .alert-critical {
        background: rgba(255, 23, 68, 0.2);
        border: 2px solid #FF1744;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 25px rgba(255, 23, 68, 0.4);
    }
    
    .alert-high {
        background: rgba(255, 107, 53, 0.2);
        border: 2px solid #FF6B35;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.4);
    }
    
    .alert-normal {
        background: rgba(0, 229, 255, 0.2);
        border: 2px solid #00E5FF;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(45deg, #FF1744, #FF6B35) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 0 25px rgba(255, 23, 68, 0.6) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(26, 26, 26, 0.95) !important;
        border-right: 2px solid #FF1744 !important;
    }
    
    /* Select boxes і inputs */
    .stSelectbox > div > div, 
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background: rgba(40, 40, 40, 0.8) !important;
        border: 2px solid #FF1744 !important;
        border-radius: 8px !important;
        color: #F8F9FA !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #00E5FF !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(40, 40, 40, 0.8) !important;
        border-radius: 10px !important;
        padding: 8px !important;
        border: 2px solid #FF1744 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-family: 'Courier New', monospace !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 23, 68, 0.1) !important;
        color: #FF1744 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255, 23, 68, 0.3) !important;
        border: 1px solid #FF1744 !important;
        color: #F8F9FA !important;
        box-shadow: 0 0 15px rgba(255, 23, 68, 0.4) !important;
    }
    
    /* Монітор-стиль текст */
    .monitor-text {
        font-family: 'Courier New', monospace !important;
        color: #00E5FF !important;
        text-shadow: 0 0 5px rgba(0, 229, 255, 0.6) !important;
        letter-spacing: 1px !important;
    }
    
    /* Grid overlay - простий */
    .grid-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(255, 23, 68, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 229, 255, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 26, 0.8);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF1744, #FF6B35);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #FF4081, #FF6B35);
    }
    
    /* Датафрейми */
    .stDataFrame {
        background: rgba(40, 40, 40, 0.8) !important;
        border-radius: 8px !important;
        border: 2px solid #FF1744 !important;
    }
    
    /* Контейнери */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Метрики значення */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #F8F9FA !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #00E5FF !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #FF6B35 !important;
        font-family: 'Courier New', monospace !important;
    }
</style>
"""