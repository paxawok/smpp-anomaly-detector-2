import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
import logging

# Імпорт теми
from cyber_japan_theme import get_cyber_japan_css

# Приховати попередження
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Конфігурація Streamlit
st.set_page_config(
    page_title="SMPP Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Застосування теми
st.markdown(get_cyber_japan_css(), unsafe_allow_html=True)

class OptimizedDashboard:
    def __init__(self):
        self.db_path = "data/db/smpp.sqlite"
        self.init_session_state()
        
    def init_session_state(self):
        """Ініціалізація session state"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'cached_data' not in st.session_state:
            st.session_state.cached_data = {}
            
    def get_connection(self):
        """Створення з'єднання з БД"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    @st.cache_data(ttl=300)  # Кешування на 5 хвилин
    def load_summary_stats(_self, time_filter="24h"):
        """Завантаження тільки основної статистики"""
        time_conditions = {
            "1h": "datetime('now', '-1 hour')",
            "24h": "datetime('now', '-1 day')", 
            "7d": "datetime('now', '-7 days')",
            "30d": "datetime('now', '-30 days')"
        }
        
        time_condition = time_conditions.get(time_filter, time_conditions["24h"])
        
        query = f"""
        SELECT 
            COUNT(*) as total_messages,
            SUM(CASE WHEN aa.is_anomaly = 1 THEN 1 ELSE 0 END) as total_anomalies,
            SUM(CASE WHEN aa.risk_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_alerts,
            AVG(aa.confidence_level) as avg_confidence,
            AVG(aa.processing_time_ms) as avg_processing_time
        FROM anomaly_analysis aa
        WHERE aa.timestamp > {time_condition}
        """
        
        try:
            with _self.get_connection() as conn:
                result = pd.read_sql_query(query, conn)
                return result.iloc[0] if not result.empty else None
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            return None
    
    @st.cache_data(ttl=600)  # Кешування на 10 хвилин
    def load_hourly_data(_self, time_filter="24h"):
        """Завантаження даних по годинах"""
        time_conditions = {
            "24h": "datetime('now', '-1 day')",
            "7d": "datetime('now', '-7 days')",
            "30d": "datetime('now', '-30 days')"
        }
        
        time_condition = time_conditions.get(time_filter, time_conditions["24h"])
        
        query = f"""
        SELECT 
            strftime('%H', aa.timestamp) as hour,
            COUNT(*) as total_count,
            SUM(CASE WHEN aa.is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
        FROM anomaly_analysis aa
        WHERE aa.timestamp > {time_condition}
        GROUP BY strftime('%H', aa.timestamp)
        ORDER BY hour
        """
        
        try:
            with _self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)
    def load_daily_data(_self, time_filter="30d"):
        """Завантаження даних по днях"""
        time_conditions = {
            "7d": "datetime('now', '-7 days')",
            "30d": "datetime('now', '-30 days')"
        }
        
        time_condition = time_conditions.get(time_filter, time_conditions["30d"])
        
        query = f"""
        SELECT 
            date(aa.timestamp) as date,
            COUNT(*) as total_count,
            SUM(CASE WHEN aa.is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count
        FROM anomaly_analysis aa
        WHERE aa.timestamp > {time_condition}
        GROUP BY date(aa.timestamp)
        ORDER BY date
        """
        
        try:
            with _self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)
    def load_category_data(_self, time_filter="24h"):
        """Завантаження даних по категоріях"""
        time_conditions = {
            "1h": "datetime('now', '-1 hour')",
            "24h": "datetime('now', '-1 day')",
            "7d": "datetime('now', '-7 days')",
            "30d": "datetime('now', '-30 days')"
        }
        
        time_condition = time_conditions.get(time_filter, time_conditions["24h"])
        
        query = f"""
        SELECT 
            sm.category,
            COUNT(*) as total_count,
            SUM(CASE WHEN aa.is_anomaly = 1 THEN 1 ELSE 0 END) as anomaly_count,
            ROUND(AVG(aa.final_anomaly_score), 3) as avg_score
        FROM anomaly_analysis aa
        JOIN smpp_messages sm ON aa.message_id = sm.id
        WHERE aa.timestamp > {time_condition}
        GROUP BY sm.category
        ORDER BY anomaly_count DESC
        LIMIT 10
        """
        
        try:
            with _self.get_connection() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def load_recent_alerts(_self, limit=20):
        """Завантаження останніх алертів"""
        query = f"""
        SELECT 
            aa.timestamp,
            aa.risk_level,
            aa.final_anomaly_score,
            aa.is_anomaly,
            sm.source_addr,
            sm.dest_addr,
            SUBSTR(sm.message_text, 1, 50) as message
        FROM anomaly_analysis aa
        JOIN smpp_messages sm ON aa.message_id = sm.id
        WHERE aa.is_anomaly = 1
        ORDER BY aa.timestamp DESC
        LIMIT {limit}
        """
        
        try:
            with _self.get_connection() as conn:
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
                <p style="color: #00E5FF; font-family: 'Courier New', monospace; font-size: 1.1em;">
                    STATUS: <span style="color: #00E5FF;">● ONLINE</span> | 
                    TIME: <span style="color: #FF1744;">{}</span> | 
                    THREAT LEVEL: <span style="color: #FF6B35;">ELEVATED</span>
                </p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    def render_key_metrics(self, stats):
        """Ключові метрики"""
        if stats is None:
            total_messages = 0
            total_anomalies = 0
            critical_alerts = 0
            avg_confidence = 0
        else:
            total_messages = int(stats['total_messages'])
            total_anomalies = int(stats['total_anomalies'])
            critical_alerts = int(stats['critical_alerts'])
            avg_confidence = float(stats['avg_confidence']) * 100 if stats['avg_confidence'] else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 ANALYZED MESSAGES",
                value=f"{total_messages:,}",
                delta=f"Total analyzed"
            )
        
        with col2:
            anomaly_rate = (total_anomalies / max(total_messages, 1)) * 100
            st.metric(
                label="⚠️ ANOMALY RATE", 
                value=f"{anomaly_rate:.2f}%",
                delta=f"{total_anomalies} anomalies"
            )
        
        with col3:
            st.metric(
                label="🚨 CRITICAL THREATS",
                value=critical_alerts,
                delta="High priority"
            )
        
        with col4:
            st.metric(
                label="🎯 AVG CONFIDENCE",
                value=f"{avg_confidence:.1f}%",
                delta="Model accuracy"
            )
    
    def render_hourly_chart(self, df):
        """Графік по годинах"""
        st.markdown("### 🕐 HOURLY ANALYSIS")
        
        if not df.empty:
            df['hour'] = df['hour'].astype(int)
            df['anomaly_rate'] = (df['anomaly_count'] / df['total_count'] * 100).round(2)
            
            fig = go.Figure()
            
            # Загальна кількість
            fig.add_trace(go.Bar(
                x=df['hour'],
                y=df['total_count'],
                name='Total Messages',
                marker_color='#00E5FF',
                opacity=0.7
            ))
            
            # Аномалії
            fig.add_trace(go.Bar(
                x=df['hour'],
                y=df['anomaly_count'],
                name='Anomalies',
                marker_color='#FF1744',
                opacity=0.9
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(10, 10, 10, 0)',
                paper_bgcolor='rgba(10, 10, 10, 0)',
                font=dict(color='#00E5FF', family='Courier New'),
                height=400,
                xaxis=dict(
                    title='HOUR OF DAY',
                    showgrid=True,
                    gridcolor='rgba(255, 23, 68, 0.2)',
                    tickfont=dict(color='#F8F9FA')
                ),
                yaxis=dict(
                    title='MESSAGE COUNT',
                    showgrid=True,
                    gridcolor='rgba(0, 229, 255, 0.2)',
                    tickfont=dict(color='#F8F9FA')
                ),
                legend=dict(
                    bgcolor='rgba(40, 40, 40, 0.8)',
                    bordercolor='#FF1744',
                    borderwidth=1,
                    font=dict(color='#F8F9FA')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No hourly data available")
    
    def render_daily_chart(self, df):
        """Графік по днях"""
        st.markdown("### 📅 DAILY TRENDS")
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['anomaly_rate'] = (df['anomaly_count'] / df['total_count'] * 100).round(2)
            
            fig = go.Figure()
            
            # Лінія аномалій
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['anomaly_count'],
                name='Daily Anomalies',
                line=dict(color='#FF1744', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 23, 68, 0.2)'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(10, 10, 10, 0)',
                paper_bgcolor='rgba(10, 10, 10, 0)',
                font=dict(color='#00E5FF', family='Courier New'),
                height=300,
                xaxis=dict(
                    title='DATE',
                    showgrid=True,
                    gridcolor='rgba(0, 229, 255, 0.2)',
                    tickfont=dict(color='#F8F9FA')
                ),
                yaxis=dict(
                    title='ANOMALY COUNT',
                    showgrid=True,
                    gridcolor='rgba(255, 23, 68, 0.2)',
                    tickfont=dict(color='#F8F9FA')
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No daily data available")
    
    def render_category_analysis(self, df):
        """Аналіз по категоріях"""
        st.markdown("### 📊 CATEGORY ANALYSIS")
        
        if not df.empty:
            df['anomaly_rate'] = (df['anomaly_count'] / df['total_count'] * 100).round(2)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=df['category'],
                    x=df['anomaly_rate'],
                    orientation='h',
                    marker=dict(
                        color=df['anomaly_rate'],
                        colorscale=[[0, '#00E5FF'], [0.5, '#FF6B35'], [1, '#FF1744']],
                        colorbar=dict(title="Anomaly Rate %")
                    ),
                    text=df['anomaly_rate'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                plot_bgcolor='rgba(10, 10, 10, 0)',
                paper_bgcolor='rgba(10, 10, 10, 0)',
                font=dict(color='#00E5FF', family='Courier New'),
                height=400,
                xaxis=dict(
                    title='ANOMALY RATE (%)',
                    showgrid=True,
                    gridcolor='rgba(0, 229, 255, 0.2)',
                    tickfont=dict(color='#F8F9FA')
                ),
                yaxis=dict(
                    title='CATEGORY',
                    tickfont=dict(color='#F8F9FA')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 No category data available")
    
    def render_recent_alerts(self, df):
        """Останні алерти"""
        st.markdown("### 🚨 RECENT ALERTS")
        
        if not df.empty:
            for idx, alert in df.iterrows():
                risk_level = alert['risk_level']
                
                if risk_level == 'CRITICAL':
                    alert_class = "alert-critical"
                elif risk_level == 'HIGH':
                    alert_class = "alert-high"
                else:
                    alert_class = "alert-normal"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #00E5FF; font-family: 'Courier New', monospace;">
                                [{alert['timestamp']}] {alert['source_addr']} → {alert['dest_addr']}
                            </strong>
                            <br>
                            <span style="font-family: 'Courier New', monospace; font-size: 0.9em; color: #F8F9FA;">
                                MSG: {alert['message']}... | SCORE: {alert['final_anomaly_score']:.3f}
                            </span>
                        </div>
                        <div>
                            <span style="background: #FF1744; padding: 8px 15px; border-radius: 8px; 
                                    font-family: 'Courier New', monospace; font-weight: bold; color: white;">
                                {risk_level}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🔍 No recent alerts")
    
    def render_sidebar(self):
        """Бокова панель"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #FF1744; font-family: 'Courier New', monospace;">⚙️ CONTROL PANEL</h2>
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MODELS", "2", delta="ACTIVE")
            with col2:
                st.metric("UPTIME", "99.9%", delta="STABLE")
            
            # Кнопки управління
            st.markdown("### 🎮 ACTIONS")
            
            if st.button("🔄 REFRESH DATA", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("📥 EXPORT REPORT", use_container_width=True):
                st.info("📊 Report generation...")
            
            # Інформація про оновлення
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; font-size: 0.9em;">
                <span class="monitor-text">
                    LAST UPDATE: {st.session_state.last_refresh.strftime('%H:%M:%S')}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            return time_filter
    
    def run(self):
        """Головна функція запуску"""
        # Заголовок
        self.render_header()
        
        # Sidebar і отримання фільтра часу
        time_filter = self.render_sidebar()
        
        st.markdown("---")
        
        # Завантаження основної статистики
        stats = self.load_summary_stats(time_filter)
        
        # Ключові метрики
        self.render_key_metrics(stats)
        
        st.markdown("---")
        
        # Графіки в дві колонки
        col1, col2 = st.columns(2)
        
        with col1:
            # Почасовий аналіз
            hourly_data = self.load_hourly_data(time_filter)
            self.render_hourly_chart(hourly_data)
        
        with col2:
            # Аналіз по категоріях
            category_data = self.load_category_data(time_filter)
            self.render_category_analysis(category_data)
        
        # Денний тренд (якщо період > 1 день)
        if time_filter in ["7d", "30d"]:
            st.markdown("---")
            daily_data = self.load_daily_data(time_filter)
            self.render_daily_chart(daily_data)
        
        st.markdown("---")
        
        # Останні алерти
        recent_alerts = self.load_recent_alerts(limit=10)
        self.render_recent_alerts(recent_alerts)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p class="monitor-text" style="font-size: 1em;">
                SMPP ANOMALY DETECTION SYSTEM v2.0 | OPTIMIZED VERSION | 
                <span style="color: #FF1744;">● SYSTEM OPERATIONAL</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.last_refresh = datetime.now()


def main():
    """Точка входу"""
    try:
        dashboard = OptimizedDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: {e}")
        st.info("Please check database connection at: data/db/smpp.sqlite")
        
        with st.expander("🔧 DEBUG INFORMATION"):
            st.write("**Expected DB Path:** data/db/smpp.sqlite")
            st.write("**Current Working Directory:**", os.getcwd())
            
            db_path = Path("data/db/smpp.sqlite")
            st.write("**Database Exists:**", db_path.exists())
            
            if db_path.exists():
                st.write("**Database Size:**", f"{db_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()