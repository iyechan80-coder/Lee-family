import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.2", layout="wide", page_icon="💎")

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

# [라이브러리 로드 안전 장치]
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

class QuantLabEngine:
    def __init__(self):
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
            st.warning("⚠️ 'vaderSentiment' 라이브러리가 설치되지 않았습니다. 감성 분석 기능이 비활성화됩니다. (설치: `pip install vaderSentiment`)")

    def _fetch_with_retry(self, ticker, period="3y", retries=3):
        """네트워크 불안정 대비 재시도 로직"""
        for i in range(retries):
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    return df
            except Exception as e:
                time.sleep(1)
        return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        """주가, 매크로, 뉴스 감성 데이터 통합 수집"""
        
        # 1. 타겟 주가 데이터
        df = _self._fetch_with_retry(ticker, period)
        if df.empty:
            return None

        # 2. 매크로 데이터 (Phase 2: VIX, 10년물 금리, 환율)
        # 3y 전체를 가져오되, 결측치는 전날 데이터로 채움
        macro_tickers = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        for m_ticker, col_name in macro_tickers.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                # 인덱스 시간대 통일 (Date만 남김)
                m_df.index = m_df.index.date
                # 메인 데이터프레임에 종가(Close)만 병합
                temp_series = m_df['Close']
                temp_series.name = col_name
                # 인덱스 기준 병합 (왼쪽 조인)
                df.index = df.index.date
                df = df.join(temp_series)

        # 3. 뉴스 데이터 및 감성 분석 (Phase 3)
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            sentiment_data = []
            if news and _self.analyzer: # 분석기가 있을 때만 실행
                for n in news:
                    title = n.get('title', '')
                    # publish time이 없는 경우 방지
                    pub_ts = n.get('providerPublishTime', time.time())
                    pub_time = datetime.datetime.fromtimestamp(pub_ts)
                    
                    # Vader 감성 분석
                    score = _self.analyzer.polarity_scores(title)['compound']
                    sentiment_data.append({'Date': pub_time.date(), 'Sentiment': score})
                
                if sentiment_data:
                    sent_df = pd.DataFrame(sentiment_data).groupby('Date').mean()
                    df = df.join(sent_df)
                else:
                    df['Sentiment'] = 0.0
            else:
                df['Sentiment'] = 0.0
                
        except Exception as e:
            # st.error(f"News fetch error: {e}") # 사용자에게 불필요한 에러 노출 최소화
            df['Sentiment'] = 0.0

        # 결측치 처리 (주말 뉴스 등은 0으로, 매크로는 전날 값으로)
        if 'Sentiment' not in df.columns:
             df['Sentiment'] = 0.0
             
        df['Sentiment'] = df['Sentiment'].fillna(0)
        df = df.ffill().bfill() # 매크로 데이터 채우기
        
        return df

    def calculate_indicators(self, df):
        """기술적 지표 계산 (BB, RSI, MA)"""
        # 데이터프레임 복사본 생성 (SettingWithCopyWarning 방지)
        df = df.copy()
        
        # MA
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Bollinger Bands
        std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (std * 2)
        df['BB_Low'] = df['MA20'] - (std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Division by zero 방지
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50) # 초기값 중립 처리
        
        return df

    def plot_dashboard(self, df, ticker):
        """4단 통합 차트 시각화 (Price, Vol, RSI, Sentiment)"""
        
        # 최근 데이터만 슬라이싱 (보기에 너무 길면 최근 1년 등으로 조정 가능하나, 여기선 전체)
        # 캔들차트 색상 설정을 위한 로직은 Plotly 내부 기능 사용
        
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"{ticker} Price Action & BB", "Volume", "RSI (14)", "News Sentiment Impact")
        )

        # 1. Price + BB + MA
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='#FFFFFF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20", line=dict(color='orange', width=1)), row=1, col=1)

        # 2. Volume
        colors_vol = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors_vol), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#00F0FF', width=1.5)), row=3, col=1)
        fig.add_trace(go.HorizontalLine(y=70, line_dash="dash", line_color="red"), row=3, col=1)
        fig.add_trace(go.HorizontalLine(y=30, line_dash="dash", line_color="green"), row=3, col=1)

        # 4. Sentiment Score
        # 감성 점수가 0인(뉴스 없는) 날은 투명하게 하거나 색을 옅게 처리
        sent_colors = ['#FF4B4B' if x < -0.05 else '#00FF7F' if x > 0.05 else 'gray' for x in df['Sentiment']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color=sent_colors), row=4, col=1)

        # Layout Update
        fig.update_layout(height=1000, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [UI Layout]
st.title("💎 Wonju AI Quant Lab (v6.2)")
st.caption("Phase 3: Sentiment Analysis Integration & Macro Tracking")

with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker = st.text_input("Ticker Symbol", value="TSLA").upper()
    period = st.selectbox("Analysis Period", ["6mo", "1y", "3y", "5y"], index=1)
    st.markdown("---")
    st.info("💡 **Tip:** 뉴스가 드문 종목은 감성 점수가 0으로 표시됩니다.")
    if not HAS_VADER:
        st.error("⚠️ 감성 분석 라이브러리(vaderSentiment) 미설치됨. 기능 제한.")

if st.button("🚀 Run Analysis", type="primary"):
    engine = QuantLabEngine()
    
    with st.spinner(f'Analyzing {ticker} with Macro & Sentiment Data...'):
        # 데이터 수집
        raw_data = engine.fetch_market_data(ticker, period)
        
        if raw_data is None or raw_data.empty:
            st.error(f"'{ticker}'에 대한 데이터를 찾을 수 없습니다.")
        else:
            # 지표 계산
            data = engine.calculate_indicators(raw_data)
            
            # 최신 데이터 추출
            last_close = data['Close'].iloc[-1]
            last_rsi = data['RSI'].iloc[-1]
            last_sent = data['Sentiment'].iloc[-1]
            
            # 매크로 데이터 (존재하는 경우만)
            last_vix = data['VIX'].iloc[-1] if 'VIX' in data.columns else 0
            last_rate = data['US_10Y'].iloc[-1] if 'US_10Y' in data.columns else 0
            
            # 상관관계 분석 (감성 vs 익일 수익률)
            # 데이터 포인트가 충분할 때만 계산
            if data['Sentiment'].abs().sum() > 0:
                corr = data['Sentiment'].corr(data['Close'].pct_change().shift(-1))
            else:
                corr = 0.0

            # 1. 상단 정보 패널 (KPI)
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Current Price", f"${last_close:.2f}", f"{data['Close'].pct_change().iloc[-1]*100:.2f}%")
            kpi2.metric("RSI (14)", f"{last_rsi:.1f}", delta_color="off")
            kpi3.metric("Sentiment Score", f"{last_sent:.2f}", help="-1.0 (Neg) ~ +1.0 (Pos)")
            kpi4.metric("US 10Y Rate", f"{last_rate:.2f}%")
            kpi5.metric("VIX Index", f"{last_vix:.2f}")

            # 2. 상관관계 분석 결과 메시지
            if abs(corr) > 0.2:
                correlation_msg = f"유의미함 ({corr:.3f})"
                msg_color = "green" if corr > 0 else "red"
                st.markdown(f"**📊 Sentiment Correlation:** <span style='color:{msg_color}'>{correlation_msg}</span> (감성지수가 주가에 영향을 줌)", unsafe_allow_html=True)
            else:
                st.markdown(f"**📊 Sentiment Correlation:** 미미함 ({corr:.3f}) (뉴스 영향력 제한적)", unsafe_allow_html=True)

            # 3. 메인 대시보드
            engine.plot_dashboard(data, ticker)
            
            # 4. 데이터 미리보기 (디버깅용)
            with st.expander("View Raw Data Frame"):
                st.dataframe(data.tail(10).style.format("{:.2f}"))
