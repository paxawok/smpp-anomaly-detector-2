import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import time
import os
from collections import deque
import json
from DatabaseManager import get_db

# Конфігурація Streamlit
st.set_page_config(
    page_title="SMPP Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стилізація
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
    }
    .alert-warning {
        background-color: #ffa500;
        color: white;
    }
    .alert-info {
        background-color: #0068c9;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 24px;
        padding-right: 24px;
    }
</style>
""", unsafe_allow_html=True)

class SMPPAnomalyDashboard:
    def __init__(self):
        # Підключення до БД
        self.db = get_db()
        
        # Ініціалізація session state
        self._init_session_state()
        
        # Завантаження моделі
        self.model_loaded = False
        self.load_latest_model()
    
    def _init_session_state(self):
        """Ініціалізація session state"""
        defaults = {
            'last_update': datetime.now(),
            'processed_count': 0,
            'anomaly_count': 0,
            'alerts': deque(maxlen=1000),
            'hourly_stats': self._init_hourly_stats(),
            'model_name': 'Unknown',
            'model_timestamp': 'Unknown'
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _init_hourly_stats(self):
        """Ініціалізація статистики по годинах"""
        return {
            'hours': list(range(24)),
            'normal': [0] * 24,
            'anomaly': [0] * 24
        }
    
    def load_latest_model(self):
        """Завантаження останньої моделі"""
        try:
            model_dir = 'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                return
                
            model_files = [f for f in os.listdir(model_dir) 
                          if f.startswith('isolation_forest_ensemble_') and f.endswith('.pkl')]
            
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(model_dir, latest_model)
                self.model_data = joblib.load(model_path)
                self.model_loaded = True
                st.session_state.model_name = latest_model
                st.session_state.model_timestamp = latest_model.split('_')[-1].replace('.pkl', '')
            else:
                st.warning("No model files found in models directory")
                
        except Exception as e:
            st.error(f"Помилка завантаження моделі: {e}")
    
    def simulate_real_time_data(self):
        """Симуляція real-time даних для демонстрації"""
        categories = ['banking', 'delivery', 'otp', 'shopping', 'government']
        sources = ['PRIVAT24', 'NOVAPOSHTA', 'GOOGLE', 'ROZETKA', 'DIA']
        
        # Генерація випадкового повідомлення
        is_anomaly = np.random.random() < 0.1  # 10% аномалій
        
        if is_anomaly:
            messages = [
                "УВАГА! Ваш рахунок заблоковано. Перейдіть bit.ly/unlock",
                "Вітаємо! Ви виграли 50000 грн! Деталі: tinyurl.com/prize",
                "Термінова! Підтвердіть операцію 10000$ за посиланням bit.ly/confirm",
                "PrivatBaнk: Спроба входу з невідомого пристрою",
                "Мон0банк: Картка тимчасово заблокована"
            ]
            source = np.random.choice(sources) + "-FAKE"
            anomaly_score = np.random.uniform(0.7, 0.95)
            risk_level = "HIGH"
        else:
            messages = [
                "Зарахування 1500 грн. Баланс: 5420 грн",
                "Ваше замовлення №12345 доставлено",
                "Код підтвердження: 4829",
                "Знижка 20% на всі товари сьогодні",
                "ПриватБанк: Операція успішна"
            ]
            source = np.random.choice(sources)
            anomaly_score = np.random.uniform(0.1, 0.4)
            risk_level = "LOW"
        
        message = np.random.choice(messages)
        
        return {
            'timestamp': datetime.now(),
            'source_addr': source,
            'dest_addr': f"380{np.random.randint(50, 99)}{np.random.randint(1000000, 9999999)}",
            'message': message[:50] + "..." if len(message) > 50 else message,
            'category': np.random.choice(categories),
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'risk_level': risk_level
        }
    
    def render_header(self):
        """Відображення заголовку"""
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.title("🛡️ SMPP Anomaly Detection System")
            st.caption("Real-time monitoring of SMS traffic anomalies")
        
        with col2:
            if self.model_loaded:
                st.success(f"✅ Model: {st.session_state.model_name[:30]}...")
                st.caption(f"Version: {st.session_state.model_timestamp}")
            else:
                st.error("❌ Model not loaded")
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
    
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def get_cached_stats(_self):
        """Кешована статистика з БД"""
        try:
            return _self.db.get_statistics()
        except Exception as e:
            st.error(f"Database error: {e}")
            return {
                'total_messages': 0,
                'total_anomalies': 0,
                'total_alerts': 0,
                'messages_last_hour': 0,
                'anomaly_rate_change': 0,
                'new_alerts_last_hour': 0
            }
    
    def render_metrics(self):
        """Відображення ключових метрик"""
        stats = self.get_cached_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📨 Total Processed",
                value=f"{stats['total_messages']:,}",
                delta=f"+{stats.get('messages_last_hour', 0)} last hour"
            )
        
        with col2:
            anomaly_rate = (stats['total_anomalies'] / max(stats['total_messages'], 1)) * 100
            st.metric(
                label="⚠️ Anomaly Rate",
                value=f"{anomaly_rate:.2f}%",
                delta=f"{stats.get('anomaly_rate_change', 0):.2f}%"
            )
        
        with col3:
            try:
                active_alerts = len(self.db.get_alerts(status='new', limit=1000))
            except:
                active_alerts = 0
                
            st.metric(
                label="🚨 Active Alerts",
                value=active_alerts,
                delta=f"+{stats.get('new_alerts_last_hour', 0)} new"
            )
        
        with col4:
            uptime = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            st.metric(
                label="⏱️ Uptime",
                value=f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
            )
    
    @st.cache_data(ttl=60)
    def get_real_time_data(_self):
        """Отримання real-time даних з БД"""
        try:
            with _self.db.get_connection() as conn:
                query = """
                    SELECT 
                        strftime('%Y-%m-%d %H:%M', timestamp) as minute,
                        COUNT(*) as total,
                        SUM(CASE WHEN a.is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
                    FROM smpp_messages m
                    LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                    WHERE m.timestamp > datetime('now', '-1 hour')
                    GROUP BY minute
                    ORDER BY minute
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"Database query error: {e}")
            return pd.DataFrame()
    
    def render_real_time_chart(self):
        """Real-time графік аномалій"""
        st.subheader("📊 Real-time Anomaly Detection")
        
        df = self.get_real_time_data()
        
        if not df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df['minute']),
                y=df['total'] - df['anomalies'].fillna(0),
                name='Normal',
                line=dict(color='#00cc44', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 68, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df['minute']),
                y=df['anomalies'].fillna(0),
                name='Anomalies',
                line=dict(color='#ff4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.1)'
            ))
            
            fig.update_layout(
                height=400,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Messages per minute",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback до симульованих даних
            st.info("📡 Using simulated data (no database data available)")
            
            time_range = pd.date_range(end=datetime.now(), periods=60, freq='1min')
            normal_counts = np.random.poisson(50, 60)
            anomaly_counts = np.random.poisson(5, 60)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_range,
                y=normal_counts,
                name='Normal (Simulated)',
                line=dict(color='#00cc44', width=2, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 68, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=time_range,
                y=anomaly_counts,
                name='Anomalies (Simulated)',
                line=dict(color='#ff4444', width=2, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.1)'
            ))
            
            fig.update_layout(
                height=400,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Messages per minute",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_tabs(self):
        """Аналіз у вкладках"""
        tab1, tab2, tab3 = st.tabs(["📈 Category Analysis", "⏰ Temporal Analysis", "🔍 Detailed Analysis"])
        
        with tab1:
            self.render_category_analysis()
        
        with tab2:
            self.render_temporal_analysis()
        
        with tab3:
            self.render_detailed_analysis()
    
    def render_category_analysis(self):
        """Аналіз по категоріях"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Anomalies by Category")
            
            try:
                with self.db.get_connection() as conn:
                    query = """
                        SELECT 
                            m.category,
                            COUNT(CASE WHEN a.is_anomaly = 1 THEN 1 END) as anomaly_count,
                            COUNT(*) as total_count
                        FROM smpp_messages m
                        LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                        WHERE m.timestamp > datetime('now', '-24 hours')
                        GROUP BY m.category
                        ORDER BY anomaly_count DESC
                    """
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['anomaly_rate'] = (df['anomaly_count'] / df['total_count'] * 100).round(2)
                    
                    fig = px.bar(
                        df,
                        x='category',
                        y='anomaly_count',
                        color='anomaly_rate',
                        color_continuous_scale='Reds',
                        labels={'category': 'Category', 'anomaly_count': 'Anomaly Count'},
                        hover_data=['total_count', 'anomaly_rate']
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No category data available")
                    
            except Exception as e:
                st.error(f"Error loading category data: {e}")
        
        with col2:
            st.subheader("📊 Category Risk Matrix")
            
            # Симульована risk matrix
            categories = ['Banking', 'Delivery', 'OTP', 'Shopping', 'Government']
            risks = np.random.rand(5) * 100
            
            colors = ['#ff4444' if r > 70 else '#ffaa44' if r > 40 else '#44ff44' for r in risks]
            
            fig = go.Figure(data=[
                go.Bar(x=categories, y=risks, marker_color=colors)
            ])
            
            fig.update_layout(
                height=300,
                yaxis_title="Risk Score",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_temporal_analysis(self):
        """Часовий аналіз"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏰ Hourly Distribution")
            
            try:
                with self.db.get_connection() as conn:
                    query = """
                        SELECT 
                            CAST(strftime('%H', timestamp) as INTEGER) as hour,
                            COUNT(*) as total,
                            SUM(CASE WHEN a.is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
                        FROM smpp_messages m
                        LEFT JOIN anomaly_analysis a ON m.id = a.message_id
                        WHERE m.timestamp > datetime('now', '-24 hours')
                        GROUP BY hour
                        ORDER BY hour
                    """
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df['hour'],
                        y=df['total'] - df['anomalies'].fillna(0),
                        name='Normal',
                        marker_color='lightgreen'
                    ))
                    fig.add_trace(go.Bar(
                        x=df['hour'],
                        y=df['anomalies'].fillna(0),
                        name='Anomaly',
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        height=300,
                        barmode='stack',
                        xaxis_title="Hour of Day",
                        yaxis_title="Message Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hourly data available")
                    
            except Exception as e:
                st.error(f"Error loading hourly data: {e}")
        
        with col2:
            st.subheader("📅 Weekly Pattern")
            
            # Симульований weekly pattern
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            normal_counts = np.random.poisson(1000, 7)
            anomaly_counts = np.random.poisson(100, 7)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=days,
                y=normal_counts,
                name='Normal',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=days,
                y=anomaly_counts,
                name='Anomalies',
                marker_color='orange'
            ))
            
            fig.update_layout(
                height=300,
                barmode='stack',
                xaxis_title="Day of Week",
                yaxis_title="Message Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_analysis(self):
        """Детальний аналіз"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Distribution")
            
            # Симульований розподіл ризиків
            risk_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            risk_counts = [np.random.randint(10, 100) for _ in risk_levels]
            colors = ['#8B0000', '#FF0000', '#FFA500', '#90EE90']
            
            fig = px.pie(
                values=risk_counts,
                names=risk_levels,
                title="Risk Level Distribution",
                color_discrete_sequence=colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Anomaly Trend")
            
            # Тренд аномалій
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            anomaly_trend = np.random.poisson(50, 30) + np.sin(np.arange(30) * 0.2) * 20
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=anomaly_trend,
                mode='lines+markers',
                name='Daily Anomalies',
                line=dict(color='red', width=2)
            ))
            
            # Додавання тренд лінії
            z = np.polyfit(range(30), anomaly_trend, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=dates,
                y=p(range(30)),
                mode='lines',
                name='Trend',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Anomaly Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_section(self):
        """Секція алертів"""
        st.subheader("🚨 Alert Management")
        
        # Фільтри для алертів
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox("Status", ["all", "new", "acknowledged", "resolved"])
        with col2:
            risk_filter = st.selectbox("Risk Level", ["all", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        with col3:
            category_filter = st.selectbox("Category", ["all", "banking", "delivery", "otp", "shopping"])
        with col4:
            limit = st.number_input("Show alerts", min_value=10, max_value=100, value=20)
        
        # Отримання та відображення алертів
        try:
            alerts = self.db.get_alerts(
                status=None if status_filter == "all" else status_filter,
                limit=limit
            )
            
            if alerts:
                self.render_alerts_table(alerts, risk_filter, category_filter)
            else:
                st.info("🎉 No alerts found matching the criteria")
                
        except Exception as e:
            st.error(f"Error loading alerts: {e}")
            # Fallback до демонстраційних даних
            st.info("📡 Showing demo alerts")
            self.render_demo_alerts()
    
    def render_alerts_table(self, alerts, risk_filter, category_filter):
        """Таблиця алертів"""
        # Фільтрація
        if risk_filter != "all":
            alerts = [a for a in alerts if a.get('risk_level') == risk_filter]
        if category_filter != "all":
            alerts = [a for a in alerts if a.get('category') == category_filter]
        
        if not alerts:
            st.info("No alerts match the selected filters")
            return
        
        # Відображення алертів
        for i, alert in enumerate(alerts[:10]):  # Показуємо тільки перші 10
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    # Іконка ризику
                    risk_icon = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠',
                        'MEDIUM': '🟡',
                        'LOW': '🟢'
                    }.get(alert.get('risk_level', 'LOW'), '⚪')
                    
                    st.write(f"{risk_icon} **{alert.get('source_addr', 'Unknown')}** → {alert.get('dest_addr', 'Unknown')}")
                    st.caption(alert.get('message_preview', 'No preview available')[:60] + "...")
                
                with col2:
                    st.write(f"**Score:** {alert.get('anomaly_score', 0):.3f}")
                    st.caption(f"Category: {alert.get('category', 'Unknown')}")
                
                with col3:
                    timestamp = alert.get('timestamp', datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
                    st.caption(f"Status: {alert.get('status', 'new')}")
                
                with col4:
                    # Кнопки дій
                    alert_id = alert.get('id', i)
                    status = alert.get('status', 'new')
                    
                    if status == 'new':
                        if st.button("✅ Acknowledge", key=f"ack_{alert_id}"):
                            try:
                                self.db.update_alert_status(alert_id, 'acknowledged', 'operator')
                                st.success("Alert acknowledged!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    elif status == 'acknowledged':
                        if st.button("🔒 Resolve", key=f"res_{alert_id}"):
                            try:
                                self.db.update_alert_status(alert_id, 'resolved', 'operator')
                                st.success("Alert resolved!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                st.divider()
    
    def render_demo_alerts(self):
        """Демонстраційні аlerти"""
        demo_alerts = [
            {
                'id': 1,
                'source_addr': 'PRIVAT-FAKE',
                'dest_addr': '380671234567',
                'message_preview': 'УВАГА! Ваш рахунок заблоковано. Перейдіть bit.ly/unlock',
                'category': 'banking',
                'risk_level': 'CRITICAL',
                'anomaly_score': 0.92,
                'status': 'new',
                'timestamp': datetime.now() - timedelta(minutes=5)
            },
            {
                'id': 2,
                'source_addr': 'UNKNOWN-NUMBER',
                'dest_addr': '380509876543',
                'message_preview': 'Вітаємо! Ви виграли iPhone 15! Деталі tinyurl.com/prize',
                'category': 'marketing',
                'risk_level': 'HIGH',
                'anomaly_score': 0.85,
                'status': 'acknowledged',
                'timestamp': datetime.now() - timedelta(minutes=15)
            }
        ]
        
        self.render_alerts_table(demo_alerts, "all", "all")
    
    def render_sidebar(self):
        """Бокова панель"""
        with st.sidebar:
            st.header("⚙️ Control Panel")
            
            # Статистика БД
            st.subheader("📊 System Stats")
            try:
                stats = self.get_cached_stats()
                st.metric("📨 Total PDUs", f"{stats.get('total_pdus', 0):,}")
                st.metric("📬 Messages", f"{stats.get('total_messages', 0):,}")
                st.metric("⚠️ Anomalies", f"{stats.get('total_anomalies', 0):,}")
                st.metric("🚨 Alerts", f"{stats.get('total_alerts', 0):,}")
            except Exception as e:
                st.error(f"Stats error: {e}")
            
            st.divider()
            
            # Налаштування оновлення
            st.subheader("🔄 Refresh Settings")
            auto_refresh = st.checkbox("Auto-refresh dashboard", value=False)
            refresh_interval = 10
            if auto_refresh:
                refresh_interval = st.slider("Interval (seconds)", 5, 60, 10)
            
            st.divider()
            
            # Інформація про модель
            st.subheader("🤖 Model Status")
            if self.model_loaded:
                st.success("✅ Model loaded")
                st.info(f"📄 {st.session_state.model_name}")
                if st.button("🔄 Reload Model"):
                    self.load_latest_model()
                    st.success("Model reloaded!")
            else:
                st.error("❌ No model loaded")
                if st.button("📂 Load Model"):
                    self.load_latest_model()
            
            st.divider()
            
            # Системна інформація
            st.subheader("💾 System Health")
            
            try:
                # Розмір БД
                db_size = os.path.getsize(self.db.db_path) / (1024 * 1024)  # MB
                st.metric("Database size", f"{db_size:.1f} MB")
                
                # Остання активність
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(timestamp) as last_activity FROM smpp_messages LIMIT 1")
                    result = cursor.fetchone()
                    if result and result['last_activity']:
                        st.info(f"🕐 Last activity: {result['last_activity']}")
                    else:
                        st.info("🕐 No recent activity")
                        
            except Exception as e:
                st.warning(f"Health check error: {e}")
            
            return auto_refresh, refresh_interval
    
    def run(self):
        """Головна функція запуску"""
        # Заголовок
        self.render_header()
        
        # Бокова панель
        auto_refresh, refresh_interval = self.render_sidebar()
        
        # Основний контент
        st.divider()
        
        # Ключові метрики
        self.render_metrics()
        
        st.divider()
        
        # Real-time графік
        self.render_real_time_chart()
        
        # Аналіз у вкладках
        self.render_analysis_tabs()
        
        st.divider()
        
        # Управління алертами
        self.render_alerts_section()
        
        # Кнопка симуляції для демонстрації
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Demo Simulation", use_container_width=True, type="primary"):
                self.run_simulation()
        
        # Автооновлення
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def run_simulation(self):
        """Демонстраційна симуляція"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(50):  # 50 iterations
            # Симуляція обробки повідомлення
            new_data = self.simulate_real_time_data()
            
            # Оновлення прогресу
            progress = (i + 1) / 50
            progress_bar.progress(progress)
            status_text.text(f"Processing message {i+1}/50: {new_data['message'][:30]}...")
            
            # Імітація обробки
            time.sleep(0.1)
            
            # Додавання до session state для демонстрації
            if 'simulation_alerts' not in st.session_state:
                st.session_state.simulation_alerts = []
            
            if new_data['is_anomaly']:
                st.session_state.simulation_alerts.append(new_data)
        
        progress_bar.empty()
        status_text.empty()
        
        # Результати симуляції
        st.success("✅ Simulation completed!")
        
        if hasattr(st.session_state, 'simulation_alerts') and st.session_state.simulation_alerts:
            st.info(f"🚨 Generated {len(st.session_state.simulation_alerts)} alerts during simulation")
            
            # Показуємо останні alerts
            with st.expander("View Generated Alerts"):
                for alert in st.session_state.simulation_alerts[-5:]:  # Last 5 alerts
                    st.write(f"**{alert['source_addr']}**: {alert['message']} (Score: {alert['anomaly_score']:.3f})")


# Функції конфігурації
def create_streamlit_config():
    """Створення .streamlit/config.toml"""
    config_content = """
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50

[browser]
gatherUsageStats = false
showErrorDetails = false

[logger]
level = "info"
"""
    
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)


def create_requirements_file():
    """Створення requirements.txt для Streamlit додатку"""
    requirements = """
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
joblib>=1.3.0
sqlite3
"""
    
    with open('requirements_dashboard.txt', 'w') as f:
        f.write(requirements)


def main():
    """Головна функція запуску додатку"""
    try:
        # Створення конфігурації
        create_streamlit_config()
        create_requirements_file()
        
        # Запуск дашборду
        dashboard = SMPPAnomalyDashboard()
        dashboard.run()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your database connection and model files.")
        
        # Показуємо інформацію для налагодження
        with st.expander("🔧 Debug Information"):
            st.write("**Current working directory:**", os.getcwd())
            st.write("**Available files:**")
            for root, dirs, files in os.walk('.'):
                level = root.replace('.', '').count(os.sep)
                indent = ' ' * 2 * level
                st.write(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files only
                    st.write(f"{subindent}{file}")
                if len(files) > 5:
                    st.write(f"{subindent}... and {len(files)-5} more files")


if __name__ == "__main__":
    main()