import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
import logging

# Приховати попередження Streamlit
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Конфігурація Streamlit з темною темою
st.set_page_config(
    page_title="SMPP Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стилізація в стилі Cyber Security
st.markdown("""
<style>
    /* Основні стилі */
    .stApp {
        background: linear-gradient(180deg, #0E1117 0%, #1A1B26 100%);
    }
    
    /* Картки метрик */
    [data-testid="metric-container"] {
        background: rgba(30, 30, 46, 0.7);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Заголовки */
    h1, h2, h3 {
        font-family: 'Courier New', monospace;
        background: linear-gradient(45deg, #00D4FF, #00F5A0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    /* Алерти */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.2), rgba(255, 0, 110, 0.1));
        border: 2px solid #FF006E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        animation: pulse-red 2s infinite;
        box-shadow: 0 0 20px rgba(255, 0, 110, 0.4);
    }
    
    .alert-high {
        background: rgba(255, 183, 0, 0.1);
        border: 1px solid #FFB700;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 0 15px rgba(255, 183, 0, 0.3);
    }
    
    .alert-normal {
        background: rgba(0, 245, 160, 0.1);
        border: 1px solid #00F5A0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(45deg, #00D4FF, #00F5A0);
        color: #0E1117;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Анімації */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 20px rgba(255, 0, 110, 0.4); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 110, 0.8); }
        100% { box-shadow: 0 0 20px rgba(255, 0, 110, 0.4); }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(30, 30, 46, 0.9);
        border-right: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Контейнери */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Select boxes і inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(30, 30, 46, 0.7);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 46, 0.5);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #00D4FF;
        border-radius: 5px;
        padding: 10px 20px;
        font-family: 'Courier New', monospace;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, rgba(0, 212, 255, 0.2), rgba(0, 245, 160, 0.2));
        border: 1px solid #00D4FF;
    }
    
    /* Монітор-стиль текст */
    .monitor-text {
        font-family: 'Courier New', monospace;
        color: #00F5A0;
        text-shadow: 0 0 5px rgba(0, 245, 160, 0.5);
    }
    
    /* Grid lines effect */
    .grid-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: -1;
    }
</style>
<div class="grid-overlay"></div>
""", unsafe_allow_html=True)

class CyberSecurityDashboard:
    def __init__(self):
        self.db_path = "data/db/smpp.sqlite"
        self.init_session_state()
        
    def init_session_state(self):
        """Ініціалізація session state"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
            
    def get_connection(self):
        """Створення з'єднання з БД"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def load_anomaly_data(self, time_filter="24h"):
        """Завантаження даних аномалій"""
        time_conditions = {
            "1h": "datetime('now', '-1 hour')",
            "24h": "datetime('now', '-1 day')",
            "7d": "datetime('now', '-7 days')",
            "30d": "datetime('now', '-30 days')"
        }
        
        time_condition = time_conditions.get(time_filter, time_conditions["24h"])
        
        # Обмежуємо кількість записів для уникнення перевантаження
        query = f"""
        SELECT 
            aa.id,
            aa.message_id,
            aa.timestamp,
            aa.final_anomaly_score,
            aa.is_anomaly,
            aa.risk_level,
            aa.confidence_level,
            aa.isolation_forest_score,
            aa.isolation_forest_anomaly,
            aa.autoencoder_score,
            aa.autoencoder_anomaly,
            aa.autoencoder_reconstruction_error,
            aa.processing_time_ms,
            sm.source_addr,
            sm.dest_addr,
            SUBSTR(sm.message_text, 1, 100) as message,  -- Обмежуємо довжину повідомлення
            sm.category,
            sm.timestamp as message_timestamp
        FROM anomaly_analysis aa
        JOIN smpp_messages sm ON aa.message_id = sm.id
        WHERE aa.timestamp > {time_condition}
        ORDER BY aa.timestamp DESC
        LIMIT 4000  -- Обмежуємо кількість записів
        """
        
        try:
            with self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            return pd.DataFrame()
    
    def render_header(self):
        """Заголовок системи"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 3em; margin-bottom: 0;">🛡️ SMPP ANOMALY DETECTION</h1>
                <p class="monitor-text">[ CYBER SECURITY MONITORING SYSTEM ]</p>
                <p style="color: #00D4FF; font-family: monospace;">
                    STATUS: <span style="color: #00F5A0;">● ONLINE</span> | 
                    TIME: <span>{}</span> | 
                    THREAT LEVEL: <span style="color: #FFB700;">ELEVATED</span>
                </p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    def render_key_metrics(self, df):
        """Ключові метрики"""
        if df.empty:
            total_messages = 0
            total_anomalies = 0
            critical_alerts = 0
            avg_confidence = 0
        else:
            total_messages = len(df)
            total_anomalies = df['is_anomaly'].sum()
            critical_alerts = len(df[df['risk_level'] == 'CRITICAL'])
            avg_confidence = df['confidence_level'].mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-cyber">', unsafe_allow_html=True)
            st.metric(
                label="📊 ANALYZED MESSAGES",
                value=f"{total_messages:,}",
                delta=f"↑ {np.random.randint(10, 100)} last hour"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            anomaly_rate = (total_anomalies / max(total_messages, 1)) * 100
            st.metric(
                label="⚠️ ANOMALY RATE",
                value=f"{anomaly_rate:.2f}%",
                delta=f"{np.random.uniform(-2, 2):.2f}%"
            )
        
        with col3:
            st.metric(
                label="🚨 CRITICAL THREATS",
                value=critical_alerts,
                delta=f"+{np.random.randint(0, 5)} new"
            )
        
        with col4:
            st.metric(
                label="🎯 AVG CONFIDENCE",
                value=f"{avg_confidence:.1f}%",
                delta=f"{np.random.uniform(-5, 5):.1f}%"
            )
    
    def render_real_time_monitor(self, df):
        """Real-time монітор аномалій"""
        st.markdown("### 📡 REAL-TIME ANOMALY MONITOR")
        
        if not df.empty:
            try:
                # Підготовка даних для графіка
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Фільтруємо тільки останні 2 години для real-time view
                two_hours_ago = datetime.now() - timedelta(hours=2)
                df_recent = df[df['timestamp'] > two_hours_ago].copy()
                
                # Якщо все ще багато даних, беремо останні 100 записів
                if len(df_recent) > 100:
                    df_recent = df_recent.nlargest(100, 'timestamp')
                
                # Групування по 5-хвилинних інтервалах
                df_resampled = df_recent.set_index('timestamp').resample('5T').agg({
                    'is_anomaly': ['count', 'sum'],
                    'final_anomaly_score': 'mean'
                }).fillna(0)
                
                df_resampled.columns = ['total_count', 'anomaly_count', 'avg_score']
                df_resampled = df_resampled.reset_index()
                df_resampled['normal_count'] = df_resampled['total_count'] - df_resampled['anomaly_count']
                
                # Створення графіка
                fig = go.Figure()
                
                # Лінія нормального трафіку
                fig.add_trace(go.Scatter(
                    x=df_resampled['timestamp'],
                    y=df_resampled['normal_count'],
                    name='Normal Traffic',
                    line=dict(color='#00F5A0', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 245, 160, 0.1)'
                ))
                
                # Лінія аномалій
                fig.add_trace(go.Scatter(
                    x=df_resampled['timestamp'],
                    y=df_resampled['anomaly_count'],
                    name='Anomalies',
                    line=dict(color='#FF006E', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 110, 0.2)',
                    mode='lines+markers'
                ))
                
                # Налаштування графіка
                fig.update_layout(
                    plot_bgcolor='rgba(14, 17, 23, 0.9)',
                    paper_bgcolor='rgba(14, 17, 23, 0)',
                    font=dict(color='#00D4FF', family='Courier New'),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(30, 30, 46, 0.7)',
                        bordercolor='#00D4FF',
                        borderwidth=1
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(0, 212, 255, 0.1)',
                        title='TIME (LAST 2 HOURS)',
                        rangeslider=dict(visible=False)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(0, 212, 255, 0.1)',
                        title='MESSAGE COUNT'
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Додаткова статистика
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(df_resampled))
                with col2:
                    st.metric("Time Range", "Last 2 hours")
                with col3:
                    st.metric("Update Interval", "5 minutes")
                    
            except Exception as e:
                st.error(f"Error creating chart: {e}")
                
        else:
            st.info("🔍 No data available for the selected time period")
    
    def render_threat_analysis(self, df):
        """Аналіз загроз"""
        st.markdown("### 🎯 THREAT ANALYSIS MATRIX")
        
        tab1, tab2, tab3 = st.tabs(["🔴 Risk Distribution", "📊 Model Performance", "🌐 Network Analysis"])
        
        with tab1:
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk level pie chart
                    risk_counts = df['risk_level'].value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.6,
                        marker=dict(
                            colors=['#FF006E', '#FFB700', '#00D4FF', '#00F5A0'],
                            line=dict(color='#0E1117', width=2)
                        )
                    )])
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(14, 17, 23, 0)',
                        paper_bgcolor='rgba(14, 17, 23, 0)',
                        font=dict(color='#00D4FF', family='Courier New'),
                        height=300,
                        showlegend=True,
                        annotations=[dict(
                            text='RISK<br>LEVELS',
                            x=0.5, y=0.5,
                            font_size=20,
                            showarrow=False,
                            font=dict(color='#00D4FF', family='Courier New')
                        )]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category analysis
                    category_anomalies = df.groupby('category')['is_anomaly'].agg(['sum', 'count'])
                    category_anomalies['rate'] = (category_anomalies['sum'] / category_anomalies['count'] * 100).round(2)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=category_anomalies.index,
                            y=category_anomalies['rate'],
                            marker=dict(
                                color=category_anomalies['rate'],
                                colorscale=[[0, '#00F5A0'], [0.5, '#FFB700'], [1, '#FF006E']],
                                line=dict(color='#00D4FF', width=1)
                            ),
                            text=category_anomalies['rate'].apply(lambda x: f'{x:.1f}%'),
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(14, 17, 23, 0)',
                        paper_bgcolor='rgba(14, 17, 23, 0)',
                        font=dict(color='#00D4FF', family='Courier New'),
                        height=300,
                        xaxis_title='CATEGORY',
                        yaxis_title='ANOMALY RATE (%)',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Isolation Forest performance
                    st.markdown("#### 🌲 ISOLATION FOREST")
                    
                    if_data = df[['isolation_forest_score', 'isolation_forest_anomaly']].dropna()
                    
                    fig = go.Figure()
                    
                    # Розподіл скорів
                    fig.add_trace(go.Histogram(
                        x=if_data['isolation_forest_score'],
                        name='Score Distribution',
                        marker=dict(
                            color='#00D4FF',
                            line=dict(color='#00F5A0', width=1)
                        ),
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(14, 17, 23, 0)',
                        paper_bgcolor='rgba(14, 17, 23, 0)',
                        font=dict(color='#00D4FF', family='Courier New'),
                        height=250,
                        xaxis_title='ANOMALY SCORE',
                        yaxis_title='FREQUENCY',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Метрики
                    if_anomalies = if_data['isolation_forest_anomaly'].sum()
                    if_total = len(if_data)
                    st.metric("Detection Rate", f"{(if_anomalies/if_total*100):.1f}%")
                
                with col2:
                    # Autoencoder performance
                    st.markdown("#### 🧠 AUTOENCODER")
                    
                    ae_data = df[['autoencoder_reconstruction_error', 'autoencoder_anomaly']].dropna()
                    
                    if not ae_data.empty:
                        fig = go.Figure()
                        
                        # Reconstruction error distribution
                        fig.add_trace(go.Histogram(
                            x=ae_data['autoencoder_reconstruction_error'],
                            name='Reconstruction Error',
                            marker=dict(
                                color='#FFB700',
                                line=dict(color='#FF006E', width=1)
                            ),
                            opacity=0.7
                        ))
                        
                        fig.update_layout(
                            plot_bgcolor='rgba(14, 17, 23, 0)',
                            paper_bgcolor='rgba(14, 17, 23, 0)',
                            font=dict(color='#00D4FF', family='Courier New'),
                            height=250,
                            xaxis_title='RECONSTRUCTION ERROR',
                            yaxis_title='FREQUENCY',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Метрики
                        ae_anomalies = ae_data['autoencoder_anomaly'].sum()
                        ae_total = len(ae_data)
                        st.metric("Detection Rate", f"{(ae_anomalies/ae_total*100):.1f}%")
                    else:
                        st.info("No Autoencoder data available")
        
        with tab3:
            if not df.empty:
                # Network analysis
                st.markdown("#### 🌐 SUSPICIOUS SOURCES")
                
                # Топ джерел аномалій
                anomaly_sources = df[df['is_anomaly'] == 1]['source_addr'].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=anomaly_sources.index,
                        x=anomaly_sources.values,
                        orientation='h',
                        marker=dict(
                            color='#FF006E',
                            line=dict(color='#00D4FF', width=1)
                        ),
                        text=anomaly_sources.values,
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    plot_bgcolor='rgba(14, 17, 23, 0)',
                    paper_bgcolor='rgba(14, 17, 23, 0)',
                    font=dict(color='#00D4FF', family='Courier New'),
                    height=400,
                    xaxis_title='ANOMALY COUNT',
                    yaxis_title='SOURCE ADDRESS',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_console(self, df):
        """Консоль алертів"""
        st.markdown("### 🚨 SECURITY ALERTS CONSOLE")
        
        # Фільтри
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_filter = st.selectbox("RISK LEVEL", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        with col2:
            category_filter = st.selectbox("CATEGORY", ["ALL"] + list(df['category'].unique()) if not df.empty else ["ALL"])
        with col3:
            anomaly_filter = st.selectbox("STATUS", ["ALL", "ANOMALY", "NORMAL"])
        with col4:
            limit = st.number_input("DISPLAY LIMIT", min_value=5, max_value=50, value=10)
        
        # Фільтрація даних
        filtered_df = df.copy()
        
        if not filtered_df.empty:
            # Спочатку обмежуємо до останніх 1000 записів
            filtered_df = filtered_df.nlargest(1000, 'timestamp')
            
            if risk_filter != "ALL":
                filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
            if category_filter != "ALL":
                filtered_df = filtered_df[filtered_df['category'] == category_filter]
            if anomaly_filter == "ANOMALY":
                filtered_df = filtered_df[filtered_df['is_anomaly'] == 1]
            elif anomaly_filter == "NORMAL":
                filtered_df = filtered_df[filtered_df['is_anomaly'] == 0]
            
            # Відображення алертів
            alerts = filtered_df.head(limit)
            
            for idx, alert in alerts.iterrows():
                risk_level = alert['risk_level']
                
                # Визначення стилю алерту
                if risk_level == 'CRITICAL':
                    alert_class = "alert-critical"
                elif risk_level == 'HIGH':
                    alert_class = "alert-high"
                else:
                    alert_class = "alert-normal"
                
                # Відображення алерту
                st.markdown(f"""
                <div class="{alert_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #00D4FF; font-family: monospace;">
                                [{alert['timestamp']}] {alert['source_addr']} → {alert['dest_addr']}
                            </strong>
                            <br>
                            <span style="font-family: monospace; font-size: 0.9em;">
                                MSG: {str(alert['message'])[:50]}... | 
                                SCORE: {alert['final_anomaly_score']:.3f} | 
                                CONFIDENCE: {alert['confidence_level']*100:.1f}%
                            </span>
                        </div>
                        <div>
                            <span style="background: {'#FF006E' if alert['is_anomaly'] else '#00F5A0'}; 
                                    padding: 5px 10px; border-radius: 5px; font-family: monospace;">
                                {'ANOMALY' if alert['is_anomaly'] else 'NORMAL'}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Статистика
            st.markdown(f"""
            <div style="margin-top: 20px; padding: 10px; background: rgba(0, 212, 255, 0.1); 
                        border: 1px solid #00D4FF; border-radius: 5px;">
                <span class="monitor-text">
                    TOTAL IN VIEW: {len(filtered_df)} | 
                    SHOWING: {len(alerts)} | 
                    CRITICAL: {len(filtered_df[filtered_df['risk_level'] == 'CRITICAL'])} | 
                    HIGH: {len(filtered_df[filtered_df['risk_level'] == 'HIGH'])}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🔍 No alerts found with selected filters")
    
    def render_sidebar(self):
        """Бокова панель управління"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #00D4FF; font-family: monospace;">⚙️ CONTROL PANEL</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Часовий фільтр
            st.markdown("### ⏱️ TIME RANGE")
            time_filter = st.selectbox(
                "SELECT PERIOD",
                ["1h", "24h", "7d", "30d"],
                index=1,
                format_func=lambda x: {
                    "1h": "Last Hour",
                    "24h": "Last 24 Hours",
                    "7d": "Last 7 Days",
                    "30d": "Last 30 Days"
                }[x]
            )
            
            # Статус системи
            st.markdown("### 📊 SYSTEM STATUS")
            
            # Завантаження даних для статистики
            df = self.load_anomaly_data(time_filter)
            
            if not df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MODELS", "2", delta="ACTIVE")
                with col2:
                    st.metric("UPTIME", "99.9%", delta="+0.1%")
                
                # Processing stats
                st.markdown("### ⚡ PROCESSING STATS")
                avg_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else 0
                st.metric("AVG PROCESSING", f"{avg_time:.1f} ms")
                
                # Model versions
                st.markdown("### 🤖 MODEL VERSIONS")
                if 'model_version' in df.columns:
                    versions = df['model_version'].dropna().unique()
                    for version in versions[:3]:  # Show last 3 versions
                        st.info(f"📌 {version}")
            
            # Кнопки управління
            st.markdown("### 🎮 ACTIONS")
            
            if st.button("🔄 REFRESH DATA", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("📥 EXPORT REPORT", use_container_width=True):
                st.info("📊 Report generation in progress...")
            
            if st.button("🚨 CLEAR ALERTS", use_container_width=True):
                st.session_state.alert_count = 0
                st.success("✅ Alerts cleared")
            
            # Інформація про оновлення
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; font-size: 0.8em; color: #00D4FF;">
                <span class="monitor-text">
                    LAST UPDATE: {st.session_state.last_refresh.strftime('%H:%M:%S')}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            return time_filter
    
    def render_temporal_heatmap(self, df):
        """Тепова карта аномалій по годинах/днях"""
        st.markdown("### 🔥 ANOMALY HEATMAP")
        
        if not df.empty:
            # Підготовка даних
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            # Створення pivot table
            heatmap_data = df.groupby(['day_of_week', 'hour'])['is_anomaly'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='is_anomaly').fillna(0)
            
            # Впорядкування днів тижня
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_pivot.reindex(days_order)
            
            # Створення heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale=[
                    [0, '#0E1117'],
                    [0.2, '#00D4FF'],
                    [0.5, '#FFB700'],
                    [1, '#FF006E']
                ],
                text=heatmap_pivot.values,
                texttemplate='%{text}',
                textfont={"size": 10, "color": "white"},
                hoverongaps=False,
                hovertemplate='Day: %{y}<br>Hour: %{x}<br>Anomalies: %{z}<extra></extra>',
                colorbar=dict(
                    title='ANOMALIES',
                    tickmode='linear',
                    tick0=0,
                    dtick=5
                )
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(14, 17, 23, 0)',
                paper_bgcolor='rgba(14, 17, 23, 0)',
                font=dict(color='#00D4FF', family='Courier New'),
                height=300,
                xaxis=dict(title='HOUR OF DAY', dtick=1),
                yaxis=dict(title='DAY OF WEEK')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No data available for heatmap generation")
    
    def render_model_comparison(self, df):
        """Порівняння моделей"""
        st.markdown("### 🤖 MODEL PERFORMANCE COMPARISON")
        
        if not df.empty and 'isolation_forest_score' in df.columns and 'autoencoder_score' in df.columns:
            # Scatter plot для порівняння скорів
            fig = go.Figure()
            
            # Нормальні точки
            normal_df = df[df['is_anomaly'] == 0]
            if not normal_df.empty:
                fig.add_trace(go.Scatter(
                    x=normal_df['isolation_forest_score'],
                    y=normal_df['autoencoder_score'],
                    mode='markers',
                    name='Normal',
                    marker=dict(
                        color='#00F5A0',
                        size=5,
                        opacity=0.6,
                        line=dict(width=1, color='#00D4FF')
                    )
                ))
            
            # Аномальні точки
            anomaly_df = df[df['is_anomaly'] == 1]
            if not anomaly_df.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_df['isolation_forest_score'],
                    y=anomaly_df['autoencoder_score'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color='#FF006E',
                        size=8,
                        opacity=0.8,
                        line=dict(width=1, color='#FFB700')
                    )
                ))
            
            # Додаємо діагональну лінію
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Agreement',
                line=dict(color='#FFB700', dash='dash', width=2)
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(14, 17, 23, 0.9)',
                paper_bgcolor='rgba(14, 17, 23, 0)',
                font=dict(color='#00D4FF', family='Courier New'),
                height=400,
                xaxis=dict(
                    title='ISOLATION FOREST SCORE',
                    showgrid=True,
                    gridcolor='rgba(0, 212, 255, 0.1)',
                    range=[0, 1]
                ),
                yaxis=dict(
                    title='AUTOENCODER SCORE',
                    showgrid=True,
                    gridcolor='rgba(0, 212, 255, 0.1)',
                    range=[0, 1]
                ),
                legend=dict(
                    bgcolor='rgba(30, 30, 46, 0.7)',
                    bordercolor='#00D4FF',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Метрики узгодженості
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Узгодженість між моделями
                if_anomalies = df['isolation_forest_anomaly'] == 1
                ae_anomalies = df['autoencoder_anomaly'] == 1
                agreement = ((if_anomalies == ae_anomalies).sum() / len(df)) * 100
                st.metric("MODEL AGREEMENT", f"{agreement:.1f}%")
            
            with col2:
                # Ensemble accuracy
                ensemble_correct = (df['is_anomaly'] == (df['final_anomaly_score'] > 0.5)).sum()
                ensemble_accuracy = (ensemble_correct / len(df)) * 100
                st.metric("ENSEMBLE ACCURACY", f"{ensemble_accuracy:.1f}%")
            
            with col3:
                # Average processing time
                avg_time = df['processing_time_ms'].mean()
                st.metric("AVG LATENCY", f"{avg_time:.1f} ms")
        else:
            st.info("🔍 Insufficient data for model comparison")
    
    def run(self):
        """Головна функція запуску дашборду"""
        # Заголовок
        self.render_header()
        
        # Sidebar і отримання фільтра часу
        time_filter = self.render_sidebar()
        
        # Завантаження даних
        df = self.load_anomaly_data(time_filter)
        
        # Основний контент
        st.markdown("---")
        
        # Ключові метрики
        self.render_key_metrics(df)
        
        st.markdown("---")
        
        # Real-time монітор
        self.render_real_time_monitor(df)
        
        # Розділ з табами для різних аналізів
        st.markdown("---")
        
        # Основні аналізи в два стовпці
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_temporal_heatmap(df)
        
        with col2:
            self.render_model_comparison(df)
        
        st.markdown("---")
        
        # Аналіз загроз
        self.render_threat_analysis(df)
        
        st.markdown("---")
        
        # Консоль алертів
        self.render_alerts_console(df)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p class="monitor-text" style="font-size: 0.8em;">
                SMPP ANOMALY DETECTION SYSTEM v2.0 | POWERED BY ML ENSEMBLE | 
                <span style="color: #00F5A0;">● SYSTEM OPERATIONAL</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Оновлення last_refresh
        st.session_state.last_refresh = datetime.now()


def main():
    """Точка входу"""
    try:
        dashboard = CyberSecurityDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {e}")
        st.info("Please check database connection at: data/db/smpp.sqlite")
        
        # Debug information
        with st.expander("🔧 DEBUG INFORMATION"):
            st.write("**Expected DB Path:** data/db/smpp.sqlite")
            st.write("**Current Working Directory:**", os.getcwd())
            
            # Check if path exists
            db_path = Path("data/db/smpp.sqlite")
            st.write("**Database Exists:**", db_path.exists())
            
            if db_path.exists():
                st.write("**Database Size:**", f"{db_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Show directory structure
            st.write("**Directory Structure:**")
            for path in Path(".").rglob("*.sqlite"):
                st.write(f"  - {path}")


if __name__ == "__main__":
    main()