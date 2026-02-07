import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import re

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.7", layout="wide", page_icon="💎")

# [전역 스타일 설정 - 가시성 극대화 (White Theme)]
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 가독성 */
    .main {
        background-color: #F8F9FA;
        color: #212529;
    }
    /* 메트릭 박스: 흰색 배경, 그림자 효과 */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #495057 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #212529 !important;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# [내장형 감성 분석기 (Fallback용)]
class LiteSentimentAnalyzer:
    """외부 라이브러리 없이 작동하는 키워드 기반 단순 감성 분석기"""
    def __init__(self):
        self.pos_words = {'up', 'rise', 'gain', 'bull', 'high', 'growth', 'profit', 'jump', 'surge', 'record', 'beat', 'buy', 'positive', 'good'}
        self.neg_words = {'down', 'fall', 'loss', 'bear', 'low', 'drop', 'crash', 'miss', 'risk', 'debt', 'sell', 'negative', 'concern', 'fail', 'bad'}

    def polarity_scores(self, text):
        text = str(text).lower()
        words = re.findall(r'\w+', text)
        score = 0
        for w in words:
            if w in self.pos_words: score += 1
            elif w in self.neg_words: score -= 1
        
        # 정규화 (-1 ~ 1 사이 값)
        norm_score = 0.0
        if score != 0:
            norm_score = score / (abs(score) + 1)
        return {'compound': norm_score}

# [엔진 클래스]
class QuantLabEngine:
    def __init__(self):
        # 모듈 로드 시도 (실패 시 Lite 버전 사용)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyzer_type = "Vader (Advanced)"
        except (ImportError, ModuleNotFoundError):
            self.analyzer = LiteSentimentAnalyzer()
            self.analyzer_type = "Lite (Built-in)"

    def _clean_index(self, df):
        """인덱스 타임존 제거 및 표준화 (MergeError 방지 핵심)"""
        if df.empty: return df
        # UTC 변환 후 타임존 제거 (Naive)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        df.index.name = 'Date'
        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        return df

    def _fetch_with_retry(self, ticker, period="3y", retries=3):
        for _ in range(retries):
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if not df.empty:
                    # MultiIndex 컬럼 단순화
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            if ticker in df.columns.get_level_values(1):
                                df = df.xs(ticker, level=1, axis=1)
                            else:
                                df.columns = df.columns.get_level_values(0)
                        except:
                            df.columns = df.columns.get_level_values(0)
                    return df
            except:
                time.sleep(1)
        return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        # 1. 메인 주가
        df = _self._fetch_with_retry(ticker, period)
        if df is None or df.empty: return None
        df = _self._clean_index(df)

        # 2. 매크로 데이터 병합
        macro_map = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        for m_ticker, col_name in macro_map.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                if 'Close' in m_df.columns:
                    series = m_df[['Close']].rename(columns={'Close': col_name})
                    # 인덱스 기준 안전 병합
                    df = pd.merge(df, series, left_index=True, right_index=True, how='left')

        # 3. 뉴스 감성 분석
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            sent_data = []
            
            if news:
                for n in news:
                    title = n.get('title', '')
                    pub_ts = n.get('providerPublishTime', time.time())
                    # 타임스탬프 -> 날짜 변환
                    pub_date = datetime.datetime.fromtimestamp(pub_ts).date() 
                    pub_dt = pd.Timestamp(pub_date) # Timestamp 객체로 변환
                    
                    score = _self.analyzer.polarity_scores(title)['compound']
                    sent_data.append({'Date': pub_dt, 'Sentiment': score})
                
                if sent_data:
                    sdf = pd.DataFrame(sent_data)
                    sdf = sdf.groupby('Date')[['Sentiment']].mean()
                    # 인덱스 타입 맞추기
                    sdf.index = pd.to_datetime(sdf.index) 
                    df = pd.merge(df, sdf, left_index=True, right_index=True, how='left')
        except Exception:
            pass # 뉴스 에러 무시

        if 'Sentiment' not in df.columns: df['Sentiment'] = 0.0
        df['Sentiment'] = df['Sentiment'].fillna(0)
        df = df.ffill().bfill()
        return df

    def calculate_indicators(self, df):
        df = df.copy()
        # MA & BB
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (2 * std)
        df['BB_Low'] = df['MA20'] - (2 * std)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        return df

    def run_backtest(self, df, rsi_buy, rsi_sell):
        """[복구됨] 사용자 설정 RSI 기반 백테스팅"""
        df = df.copy()
        df['Signal'] = 0
        # 사용자 입력값(rsi_buy, rsi_sell) 적용
        df.loc[df['RSI'] < rsi_buy, 'Signal'] = 1  # 매수
        df.loc[df['RSI'] > rsi_sell, 'Signal'] = -1 # 매도
        
        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
        df['Position'] = df['Position'].clip(lower=0) 
        
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
        
        cum_market = (1 + df['Market_Return']).cumprod().iloc[-1] - 1
        cum_strategy = (1 + df['Strategy_Return']).cumprod().iloc[-1] - 1
        
        return cum_market, cum_strategy

    def generate_gems_pack(self, df, ticker, m_ret, s_ret):
        """tabulate 의존성 제거 및 데이터 팩 생성"""
        last = df.iloc[-1]
        corr_sent = df['Close'].corr(df['Sentiment'])
        
        # .to_string() 사용으로 ImportError 방지
        recent_trend = df[['Close', 'RSI', 'Sentiment']].tail(5).to_string()
        
        report = f"""
### 💎 Wonju Quant Lab: Deep Dive Report ({ticker})
**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

#### 1. Technical & Strategy Summary
- **Current Price:** ${last['Close']:.2f}
- **RSI (14):** {last['RSI']:.2f}
- **Strategy Return:** {s_ret*100:.2f}% (vs Buy&Hold: {m_ret*100:.2f}%)
- **Bollinger Band:** {'Above Upper' if last['Close'] > last['BB_High'] else 'Below Lower' if last['Close'] < last['BB_Low'] else 'Inside'}

#### 2. Macro & Sentiment
- **Sentiment Score:** {last['Sentiment']:.3f} (Corr: {corr_sent:.3f})
- **VIX:** {last.get('VIX', 0):.2f}
- **US 10Y:** {last.get('US_10Y', 0):.2f}%
- **USD/KRW:** {last.get('USD_KRW', 0):.2f}

#### 3. Recent Trend (Last 5 Days)
{recent_trend}

---
*Prompt for Gems: "Analyze this data. Check for RSI divergences and suggest a strategy based on VIX levels."*
"""
        return report

    def plot_dashboard(self, df, ticker, rsi_buy, rsi_sell):
        """가시성 개선 차트 + RSI 기준선 연동"""
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.06, 
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"{ticker} Price & BB", "Volume", f"RSI (Buy < {rsi_buy}, Sell > {rsi_sell})", "Sentiment & Macro")
        )

        # 1. Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='black', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20", line=dict(color='orange', width=1)), row=1, col=1)

        # 2. Volume
        colors = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple', width=1.5)), row=3, col=1)
        # 사용자 설정 기준선 표시
        fig.add_hline(y=rsi_sell, line_dash="dash", line_color="red", annotation_text="Sell", row=3, col=1)
        fig.add_hline(y=rsi_buy, line_dash="dash", line_color="green", annotation_text="Buy", row=3, col=1)

        # 4. Sentiment vs VIX
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color='blue', opacity=0.5), row=4, col=1)
        if 'VIX' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX", line=dict(color='red', width=1), yaxis='y2'), row=4, col=1)

        fig.update_layout(
            height=1000, 
            template="plotly_white", 
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# [UI 실행]
st.title("💎 Wonju AI Quant Lab (v6.7)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker = st.text_input("Ticker", "TSLA").upper()
    period = st.selectbox("Period", ["1y", "3y", "5y"], index=1)
    
    st.markdown("---")
    st.subheader("🛠️ Strategy Settings")
    rsi_buy = st.slider("RSI Buy Threshold", 10, 40, 30) # 매수 기준 (기본 30)
    rsi_sell = st.slider("RSI Sell Threshold", 60, 90, 70) # 매도 기준 (기본 70)

engine = QuantLabEngine()
st.caption(f"Engine Status: {engine.analyzer_type} | Mode: Interactive Backtest")

if st.button("🚀 Run Full Analysis", type="primary"):
    with st.spinner("Processing Market Data & Simulating Strategy..."):
        df = engine.fetch_market_data(ticker, period)
        
        if df is not None and not df.empty:
            df = engine.calculate_indicators(df)
            
            # 동적 백테스팅 수행 (슬라이더 값 적용)
            m_ret, s_ret = engine.run_backtest(df, rsi_buy, rsi_sell)
            
            # KPI 출력
            last = df.iloc[-1]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Price", f"${last['Close']:.2f}", f"{(last['Close']/df.iloc[-2]['Close']-1)*100:.1f}%")
            k2.metric("RSI Strategy", f"{s_ret*100:.1f}%", f"vs Mkt {m_ret*100:.1f}%")
            k3.metric("Sentiment", f"{last['Sentiment']:.2f}")
            k4.metric("USD/KRW", f"₩{last.get('USD_KRW', 0):,.0f}")
            k5.metric("VIX", f"{last.get('VIX', 0):.2f}")
            
            # 차트 (사용자 설정 기준선 적용)
            engine.plot_dashboard(df, ticker, rsi_buy, rsi_sell)
            
            # Gems Pack
            st.markdown("---")
            st.subheader("📦 Gems Data Pack")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text_area("Copy this for Gems/LLM:", engine.generate_gems_pack(df, ticker, m_ret, s_ret), height=250)
            with c2:
                st.success("✅ Analysis Complete")
        else:
            st.error("데이터 수집 실패. 티커를 확인하세요.")
