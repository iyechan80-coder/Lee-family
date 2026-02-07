import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.4", layout="wide", page_icon="💎")

# [전역 스타일 설정]
st.markdown("""
    <style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# [라이브러리 로드 안전 장치 - 수정됨]
# 모듈이 없어도 코드가 멈추지 않도록 전역 변수로 플래그 설정
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
                    # MultiIndex 컬럼 단순화 (v0.2.x 호환)
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            # Ticker가 레벨에 있다면 해당 Ticker만 추출
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
        """인덱스 표준화 (Timezone 제거 및 이름 통일)"""
        if df.empty:
            return df
        # 1. Timezone 제거 (UTC -> Naive)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        # 2. 인덱스 이름 명시 (MergeError 방지)
        df.index.name = 'Date'
        # 3. 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        return df

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        """주가, 매크로, 뉴스 감성 데이터 통합 수집"""
        
        # 1. 타겟 주가 데이터
        df = _self._fetch_with_retry(ticker, period)
        if df is None or df.empty:
            return None
        
        df = _self._clean_index(df)

        # 2. 매크로 데이터 병합 (MergeError 해결을 위해 pd.merge + index 명시 사용)
        macro_tickers = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        
        for m_ticker, col_name in macro_tickers.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                
                if 'Close' in m_df.columns:
                    temp_series = m_df[['Close']].rename(columns={'Close': col_name})
                    # DataFrame끼리 병합 (가장 안전한 방법)
                    df = pd.merge(df, temp_series, left_index=True, right_index=True, how='left')

        # 3. 뉴스 데이터 및 감성 분석
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            sentiment_data = []
            # 분석기가 정상 로드되었고 뉴스가 있을 때만 실행
            if news and _self.analyzer: 
                for n in news:
                    title = n.get('title', '')
                    pub_ts = n.get('providerPublishTime', time.time())
                    # UTC로 변환
                    pub_time = datetime.datetime.fromtimestamp(pub_ts, datetime.timezone.utc)
                    
                    score = _self.analyzer.polarity_scores(title)['compound']
                    sentiment_data.append({'Date': pub_time, 'Sentiment': score})
                
                if sentiment_data:
                    sent_df = pd.DataFrame(sentiment_data)
                    # 날짜 정규화
                    sent_df['Date'] = pd.to_datetime(sent_df['Date']).dt.tz_localize(None).dt.normalize()
                    # 일별 평균 산출
                    sent_df = sent_df.groupby('Date')[['Sentiment']].mean()
                    
                    # 주가 데이터와 병합
                    df = pd.merge(df, sent_df, left_index=True, right_index=True, how='left')
                else:
                    df['Sentiment'] = 0.0
            else:
                df['Sentiment'] = 0.0
                
        except Exception:
            # 뉴스 처리 중 어떤 에러가 나도 주가 분석은 멈추지 않음
            if 'Sentiment' not in df.columns:
                df['Sentiment'] = 0.0

        # 결측치 처리
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

    def plot_dashboard(self, df, ticker):
        """4단 통합 차트 시각화 (AttributeError 수정됨)"""
        
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"{ticker} Price Action & BB", "Volume", "RSI (14)", "News Sentiment Impact")
        )

        # 1. Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='#FFFFFF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20", line=dict(color='orange', width=1)), row=1, col=1)

        # 2. Volume
        colors_vol = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors_vol), row=2, col=1)
        
        # 3. RSI - [수정됨] go.HorizontalLine 대신 add_hline 사용
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#00F0FF', width=1.5)), row=3, col=1)
        # 중요: row/col을 명시하여 서브플롯에만 선 긋기
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # 4. Sentiment
        sent_colors = ['#FF4B4B' if x < -0.05 else '#00FF7F' if x > 0.05 else 'gray' for x in df['Sentiment']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color=sent_colors), row=4, col=1)

        fig.update_layout(height=1000, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [UI Layout]
st.title("💎 Wonju AI Quant Lab (v6.4)")
st.caption("Phase 3: Sentiment Analysis Integration & Macro Tracking (Stable)")

with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker = st.text_input("Ticker Symbol", value="TSLA").upper()
    period = st.selectbox("Analysis Period", ["6mo", "1y", "3y", "5y"], index=1)
    st.markdown("---")
    if not HAS_VADER:
        st.warning("⚠️ 감성 분석 모듈 미설치. 차트는 중립(0)으로 표시됩니다.")
        st.code("pip install vaderSentiment", language="bash")

if st.button("🚀 Run Analysis", type="primary"):
    engine = QuantLabEngine()
    
    with st.spinner(f'Analyzing {ticker} with Macro & Sentiment Data...'):
        raw_data = engine.fetch_market_data(ticker, period)
        
        if raw_data is None or raw_data.empty:
            st.error(f"'{ticker}' 데이터를 불러올 수 없습니다.")
        else:
            data = engine.calculate_indicators(raw_data)
            
            last_close = data['Close'].iloc[-1]
            last_rsi = data['RSI'].iloc[-1]
            last_sent = data['Sentiment'].iloc[-1]
            last_vix = data['VIX'].iloc[-1] if 'VIX' in data.columns else 0
            last_rate = data['US_10Y'].iloc[-1] if 'US_10Y' in data.columns else 0
            
            if data['Sentiment'].abs().sum() > 0:
                corr = data['Sentiment'].corr(data['Close'].pct_change().shift(-1))
            else:
                corr = 0.0

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Current Price", f"${last_close:.2f}", f"{data['Close'].pct_change().iloc[-1]*100:.2f}%")
            kpi2.metric("RSI (14)", f"{last_rsi:.1f}", delta_color="off")
            kpi3.metric("Sentiment Score", f"{last_sent:.2f}", help="-1.0 (Neg) ~ +1.0 (Pos)")
            kpi4.metric("US 10Y Rate", f"{last_rate:.2f}%")
            kpi5.metric("VIX Index", f"{last_vix:.2f}")

            if abs(corr) > 0.2:
                correlation_msg = f"유의미함 ({corr:.3f})"
                msg_color = "green" if corr > 0 else "red"
                st.markdown(f"**📊 Sentiment Correlation:** <span style='color:{msg_color}'>{correlation_msg}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**📊 Sentiment Correlation:** 미미함 ({corr:.3f})", unsafe_allow_html=True)

            engine.plot_dashboard(data, ticker)
            
            with st.expander("View Raw Data Frame"):
                st.dataframe(data.tail(10).style.format("{:.2f}"))
