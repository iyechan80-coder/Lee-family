import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.3", layout="wide", page_icon="💎")

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
        # 분석기가 있으면 초기화, 없으면 None (기능 비활성화)
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None

    def _fetch_with_retry(self, ticker, period="3y", retries=3):
        """네트워크 불안정 및 데이터 형식 오류 대비 재시도 로직"""
        for i in range(retries):
            try:
                # yfinance 데이터 다운로드 (auto_adjust=True로 수정 주가 확보)
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                
                if not df.empty:
                    # MultiIndex 컬럼 처리 (예: ('Close', 'TSLA') -> 'Close')
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            # Ticker 레벨이 있는 경우 해당 Ticker만 추출
                            if ticker in df.columns.get_level_values(1):
                                df = df.xs(ticker, level=1, axis=1)
                            else:
                                # 레벨 구조가 다른 경우 첫 번째 레벨(Price Type)만 사용
                                df.columns = df.columns.get_level_values(0)
                        except Exception:
                            # 예외 발생 시 강제로 첫 번째 레벨 사용
                            df.columns = df.columns.get_level_values(0)
                    return df
            except Exception as e:
                time.sleep(1) # 실패 시 1초 대기 후 재시도
        return pd.DataFrame()

    def _clean_index(self, df):
        """인덱스를 표준 날짜 형식(Timezone Naive)으로 변환 및 중복 제거"""
        if df.empty:
            return df
        # 1. Datetime 변환 및 Timezone 제거 (UTC, Local 혼용 방지)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        # 2. 중복 날짜 제거 (데이터 꼬임 방지, 첫 번째 값 유지)
        df = df[~df.index.duplicated(keep='first')]
        return df

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        """주가, 매크로, 뉴스 감성 데이터 통합 수집"""
        
        # 1. 타겟 주가 데이터 확보
        df = _self._fetch_with_retry(ticker, period)
        if df.empty:
            return None
        
        # 인덱스 정리 (MergeError 방지의 핵심)
        df = _self._clean_index(df)

        # 2. 매크로 데이터 병합 (Phase 2: VIX, 10년물 금리, 환율)
        # 리소스 절약을 위해 필요한 지표만 순차적으로 호출
        macro_tickers = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        
        for m_ticker, col_name in macro_tickers.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                
                # 종가(Close) 컬럼만 추출하여 병합
                if 'Close' in m_df.columns:
                    temp_series = m_df['Close']
                    temp_series.name = col_name
                    
                    # 인덱스 기준 Left Join (주가 데이터 기준)
                    # pd.merge를 사용하여 인덱스 충돌 없이 안전하게 병합
                    df = pd.merge(df, temp_series, left_index=True, right_index=True, how='left')

        # 3. 뉴스 데이터 및 감성 분석 (Phase 3)
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            
            sentiment_data = []
            # 분석기가 있고 뉴스가 존재하는 경우에만 실행
            if news and _self.analyzer: 
                for n in news:
                    title = n.get('title', '')
                    # publish time 처리 (UTC 기준 타임스탬프 변환)
                    pub_ts = n.get('providerPublishTime', time.time())
                    pub_time = datetime.datetime.fromtimestamp(pub_ts, datetime.timezone.utc)
                    
                    # Vader 감성 분석 수행
                    score = _self.analyzer.polarity_scores(title)['compound']
                    sentiment_data.append({'Date': pub_time, 'Sentiment': score})
                
                if sentiment_data:
                    sent_df = pd.DataFrame(sentiment_data)
                    # 뉴스 날짜 정규화 (주가 데이터와 동일하게 맞춤)
                    sent_df['Date'] = pd.to_datetime(sent_df['Date']).dt.tz_localize(None).dt.normalize()
                    
                    # 같은 날짜의 뉴스는 평균 점수로 산출 (DataFrame 반환 보장)
                    sent_df = sent_df.groupby('Date')[['Sentiment']].mean()
                    
                    # 주가 데이터와 병합
                    df = pd.merge(df, sent_df, left_index=True, right_index=True, how='left')
                else:
                    df['Sentiment'] = 0.0
            else:
                df['Sentiment'] = 0.0
                
        except Exception:
            # 뉴스 데이터 처리 중 오류 발생 시, 전체 프로세스를 멈추지 않고 0으로 처리
            if 'Sentiment' not in df.columns:
                df['Sentiment'] = 0.0

        # 결측치 최종 처리
        if 'Sentiment' not in df.columns:
             df['Sentiment'] = 0.0
             
        # 감성 점수 없는 날은 0(중립)으로 채움
        df['Sentiment'] = df['Sentiment'].fillna(0)
        # 매크로 지표 결측치는 전날 데이터로 채움 (ffill -> bfill)
        df = df.ffill().bfill() 
        
        return df

    def calculate_indicators(self, df):
        """기술적 지표 계산 (BB, RSI, MA) - 벡터 연산으로 고속 처리"""
        df = df.copy()
        
        # 이동평균 (MA20)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 볼린저 밴드 (Bollinger Bands)
        std = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['MA20'] + (std * 2)
        df['BB_Low'] = df['MA20'] - (std * 2)
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        loss = loss.replace(0, np.nan) # 0으로 나누기 방지
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50) # 초기값은 50(중립)으로 설정
        
        return df

    def plot_dashboard(self, df, ticker):
        """4단 통합 차트 시각화 (Plotly)"""
        
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

        # 2. Volume (상승/하락에 따른 색상 구분)
        colors_vol = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors_vol), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#00F0FF', width=1.5)), row=3, col=1)
        fig.add_trace(go.HorizontalLine(y=70, line_dash="dash", line_color="red"), row=3, col=1)
        fig.add_trace(go.HorizontalLine(y=30, line_dash="dash", line_color="green"), row=3, col=1)

        # 4. Sentiment Score
        # 긍정(초록), 부정(빨강), 중립(회색) 색상 매핑
        sent_colors = ['#FF4B4B' if x < -0.05 else '#00FF7F' if x > 0.05 else 'gray' for x in df['Sentiment']]
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color=sent_colors), row=4, col=1)

        fig.update_layout(height=1000, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [UI Layout]
st.title("💎 Wonju AI Quant Lab (v6.3)")
st.caption("Phase 3: Sentiment Analysis Integration & Macro Tracking")

with st.sidebar:
    st.header("⚙️ Control Panel")
    ticker = st.text_input("Ticker Symbol", value="TSLA").upper()
    period = st.selectbox("Analysis Period", ["6mo", "1y", "3y", "5y"], index=1)
    st.markdown("---")
    st.info("💡 **Tip:** 뉴스가 드문 종목은 감성 점수가 0으로 표시됩니다.")
    if not HAS_VADER:
        st.warning("⚠️ 감성 분석 라이브러리(vaderSentiment) 미설치됨. 기능이 제한됩니다.")

if st.button("🚀 Run Analysis", type="primary"):
    engine = QuantLabEngine()
    
    with st.spinner(f'Analyzing {ticker} with Macro & Sentiment Data...'):
        raw_data = engine.fetch_market_data(ticker, period)
        
        if raw_data is None or raw_data.empty:
            st.error(f"'{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커를 확인하거나 잠시 후 다시 시도하세요.")
        else:
            data = engine.calculate_indicators(raw_data)
            
            # KPI 지표 추출
            last_close = data['Close'].iloc[-1]
            last_rsi = data['RSI'].iloc[-1]
            last_sent = data['Sentiment'].iloc[-1]
            last_vix = data['VIX'].iloc[-1] if 'VIX' in data.columns else 0
            last_rate = data['US_10Y'].iloc[-1] if 'US_10Y' in data.columns else 0
            
            # 상관관계 분석
            if data['Sentiment'].abs().sum() > 0:
                # 감성 점수와 '다음 날' 주가 등락률 간의 상관관계
                corr = data['Sentiment'].corr(data['Close'].pct_change().shift(-1))
            else:
                corr = 0.0

            # 1. KPI 대시보드
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Current Price", f"${last_close:.2f}", f"{data['Close'].pct_change().iloc[-1]*100:.2f}%")
            kpi2.metric("RSI (14)", f"{last_rsi:.1f}", delta_color="off")
            kpi3.metric("Sentiment Score", f"{last_sent:.2f}", help="-1.0 (Neg) ~ +1.0 (Pos)")
            kpi4.metric("US 10Y Rate", f"{last_rate:.2f}%")
            kpi5.metric("VIX Index", f"{last_vix:.2f}")

            # 2. 인사이트 메시지
            if abs(corr) > 0.2:
                correlation_msg = f"유의미함 ({corr:.3f})"
                msg_color = "green" if corr > 0 else "red"
                st.markdown(f"**📊 Sentiment Correlation:** <span style='color:{msg_color}'>{correlation_msg}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**📊 Sentiment Correlation:** 미미함 ({corr:.3f})", unsafe_allow_html=True)

            # 3. 차트 시각화
            engine.plot_dashboard(data, ticker)
            
            # 4. 데이터 검증용 테이블
            with st.expander("View Raw Data Frame"):
                st.dataframe(data.tail(10).style.format("{:.2f}"))
