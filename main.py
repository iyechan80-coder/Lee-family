import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.5", layout="wide", page_icon="💎")

# [전역 스타일 설정 - 가시성 개선]
st.markdown("""
    <style>
    /* 메트릭 박스 디자인 개선 */
    div[data-testid="stMetric"] {
        background-color: #262730; /* 조금 더 밝은 다크 그레이 */
        border: 1px solid #4F4F4F;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0 !important; /* 라벨은 연한 회색 */
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* 값은 흰색 강조 */
    }
    </style>
    """, unsafe_allow_html=True)

# [라이브러리 로드 안전 장치]
HAS_VADER = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except (ImportError, ModuleNotFoundError):
    HAS_VADER = False

class QuantLabEngine:
    def __init__(self):
        self.analyzer = None
        if HAS_VADER:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except Exception:
                self.analyzer = None

    def _fetch_with_retry(self, ticker, period="3y", retries=3):
        """네트워크 불안정 및 데이터 형식 오류 대비 재시도 로직"""
        for i in range(retries):
            try:
                # auto_adjust=True: 수정주가 사용
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                
                if not df.empty:
                    # MultiIndex 컬럼 단순화
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            if ticker in df.columns.get_level_values(1):
                                df = df.xs(ticker, level=1, axis=1)
                            else:
                                df.columns = df.columns.get_level_values(0)
                        except Exception:
                            df.columns = df.columns.get_level_values(0)
                    return df
            except Exception:
                time.sleep(1)
        return pd.DataFrame()

    def _clean_index(self, df):
        """인덱스 표준화"""
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = 'Date'
        df = df[~df.index.duplicated(keep='first')]
        return df

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        """주가, 매크로, 뉴스 감성 데이터 통합 수집"""
        df = _self._fetch_with_retry(ticker, period)
        if df is None or df.empty:
            return None
        
        df = _self._clean_index(df)

        # 매크로 데이터 병합
        macro_tickers = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        for m_ticker, col_name in macro_tickers.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                if 'Close' in m_df.columns:
                    temp_series = m_df[['Close']].rename(columns={'Close': col_name})
                    df = pd.merge(df, temp_series, left_index=True, right_index=True, how='left')

        # 뉴스 데이터 및 감성 분석
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            sentiment_data = []
            if news and _self.analyzer: 
                for n in news:
                    title = n.get('title', '')
                    pub_ts = n.get('providerPublishTime', time.time())
                    pub_time = datetime.datetime.fromtimestamp(pub_ts, datetime.timezone.utc)
                    score = _self.analyzer.polarity_scores(title)['compound']
                    sentiment_data.append({'Date': pub_time, 'Sentiment': score})
                
                if sentiment_data:
                    sent_df = pd.DataFrame(sentiment_data)
                    sent_df['Date'] = pd.to_datetime(sent_df['Date']).dt.tz_localize(None).dt.normalize()
                    sent_df = sent_df.groupby('Date')[['Sentiment']].mean()
                    df = pd.merge(df, sent_df, left_index=True, right_index=True, how='left')
                else:
                    df['Sentiment'] = 0.0
            else:
                df['Sentiment'] = 0.0
        except Exception:
            if 'Sentiment' not in df.columns:
                df['Sentiment'] = 0.0

        if 'Sentiment' not in df.columns:
             df['Sentiment'] = 0.0
             
        df['Sentiment'] = df['Sentiment'].fillna(0)
        df = df.ffill().bfill()
        
        return df

    def calculate_indicators(self, df):
        df = df.copy()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (std * 2)
        df['BB_Low'] = df['MA20'] - (std * 2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        return df

    def generate_data_pack(self, df, ticker):
        """Gems 심층 분석용 데이터 팩 생성"""
        last_row = df.iloc[-1]
        last_5 = df.tail(5)
        
        # 텍스트 리포트 생성
        report = f"""
[Wonju Quant Lab Analysis Data Pack]
Target: {ticker}
Date: {datetime.datetime.now().strftime('%Y-%m-%d')}

1. Market Overview (Latest)
- Price: ${last_row['Close']:.2f}
- RSI(14): {last_row['RSI']:.2f} (Status: {"Overbought" if last_row['RSI']>70 else "Oversold" if last_row['RSI']<30 else "Neutral"})
- BB Position: {"Above Upper" if last_row['Close'] > last_row['BB_High'] else "Below Lower" if last_row['Close'] < last_row['BB_Low'] else "Inside Bands"}
- Sentiment Score: {last_row['Sentiment']:.3f}

2. Macro Context
- VIX: {last_row.get('VIX', 'N/A')}
- US 10Y Rate: {last_row.get('US_10Y', 'N/A')}%

3. Recent Trend (Last 5 Days)
{last_5[['Close', 'RSI', 'Sentiment']].to_string()}
"""
        return report

    def plot_dashboard(self, df, ticker):
        """4단 통합 차트 시각화 (범례 복구 및 가시성 개선)"""
        
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"{ticker} Price Action & BB", "Volume", "RSI (14)", "News Sentiment Impact")
        )

        # 1. Price (범례 추가: showlegend=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price", line=dict(color='#FFFFFF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20 (Center)", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # 2. Volume
        colors_vol = ['#FF4B4B' if r['Open'] > r['Close'] else '#00FF7F' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors_vol, showlegend=False), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#00F0FF', width=1.5), showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # 4. Sentiment
        sent_colors = ['#FF4B4B' if x < -0.05 else '#00FF7F' if x > 0.05 else 'gray' for x in df['Sentiment']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color=sent_colors, showlegend=False), row=4, col=1)

        # Layout Update (범례 위치 조정)
        fig.update_layout(
            height=1000, 
            template="plotly_dark", 
            showlegend=True, # 범례 활성화
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# [UI Layout]
st.title("💎 Wonju AI Quant Lab (v6.5)")
st.caption("Phase 3 Complete: Visibility Improved & Data Pack Restored")

with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker = st.text_input("Ticker Symbol", value="TSLA").upper()
    period = st.selectbox("Analysis Period", ["6mo", "1y", "3y", "5y"], index=1)
    st.markdown("---")
    if not HAS_VADER:
        st.warning("⚠️ 감성 분석 모듈 미설치. (pip install vaderSentiment 필요)")

if st.button("🚀 Run Analysis", type="primary"):
    engine = QuantLabEngine()
    
    with st.spinner(f'Analyzing {ticker}...'):
        raw_data = engine.fetch_market_data(ticker, period)
        
        if raw_data is None or raw_data.empty:
            st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
        else:
            data = engine.calculate_indicators(raw_data)
            
            # KPI 데이터 추출
            last_row = data.iloc[-1]
            last_close = last_row['Close']
            prev_close = data.iloc[-2]['Close']
            chg_pct = (last_close - prev_close) / prev_close * 100
            
            # 1. KPI 대시보드 (가시성 개선된 스타일 적용)
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Current Price", f"${last_close:.2f}", f"{chg_pct:.2f}%")
            kpi2.metric("RSI (14)", f"{last_row['RSI']:.1f}", delta_color="off")
            kpi3.metric("Sentiment", f"{last_row['Sentiment']:.2f}", help="News Sentiment Score")
            kpi4.metric("US 10Y Rate", f"{last_row.get('US_10Y', 0):.2f}%")
            kpi5.metric("VIX Index", f"{last_row.get('VIX', 0):.2f}")

            # 2. 메인 차트
            engine.plot_dashboard(data, ticker)
            
            # 3. Gems 데이터 팩 (복구됨)
            st.markdown("---")
            st.subheader("📦 Gems Data Pack")
            col_pack1, col_pack2 = st.columns([3, 1])
            
            data_report = engine.generate_data_pack(data, ticker)
            with col_pack1:
                st.text_area("Analysis Context (Copy for LLM)", data_report, height=200)
            
            with col_pack2:
                st.download_button(
                    label="📥 Download CSV",
                    data=data.to_csv().encode('utf-8'),
                    file_name=f"{ticker}_quant_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.info("👆 이 데이터를 Gems에 업로드하여 심층 분석을 요청하세요.")
