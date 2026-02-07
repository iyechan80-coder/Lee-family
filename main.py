import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import re
import json

# 구글 시트 연동 라이브러리 (v6.1 복구 유지)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.10", layout="wide", page_icon="💎")

# [전역 스타일 설정 - 가시성 극대화 (White Theme)]
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; color: #212529; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #495057 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #212529 !important; font-weight: 700; }
    .gems-guide {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# [내장형 감성 분석기]
class LiteSentimentAnalyzer:
    def __init__(self):
        self.pos_words = {'up', 'rise', 'gain', 'bull', 'high', 'growth', 'profit', 'jump', 'surge', 'record', 'beat', 'buy', 'positive', 'good'}
        self.neg_words = {'down', 'fall', 'loss', 'bear', 'low', 'drop', 'crash', 'miss', 'risk', 'debt', 'sell', 'negative', 'concern', 'fail', 'bad'}
    def polarity_scores(self, text):
        text = str(text).lower()
        words = re.findall(r'\w+', text)
        score = sum(1 for w in words if w in self.pos_words) - sum(1 for w in words if w in self.neg_words)
        norm_score = score / (abs(score) + 1) if score != 0 else 0.0
        return {'compound': norm_score}

class QuantLabEngine:
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyzer_type = "Vader (Adv)"
        except:
            self.analyzer = LiteSentimentAnalyzer()
            self.analyzer_type = "Lite (Built-in)"

    def _clean_index(self, df):
        """인덱스 타임존 제거 및 표준화 (MergeError 방지)"""
        if df.empty: return df
        df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None).normalize()
        df.index.name = 'Date'
        return df[~df.index.duplicated(keep='first')]

    def _fetch_with_retry(self, ticker, period="3y", retries=2):
        for _ in range(retries):
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            df = df.xs(ticker, level=1, axis=1) if ticker in df.columns.get_level_values(1) else df.columns.get_level_values(0)
                        except:
                            df.columns = df.columns.get_level_values(0)
                    return df
            except: time.sleep(0.5)
        return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def fetch_market_data(_self, ticker, period="3y"):
        # 1. 메인 주가
        df = _self._fetch_with_retry(ticker, period)
        if df.empty: return None
        df = _self._clean_index(df)

        # 2. 매크로 데이터 병합 (환율, 금리, VIX)
        macro_map = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        for m_ticker, col in macro_map.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                if 'Close' in m_df.columns:
                    series = m_df[['Close']].rename(columns={'Close': col})
                    df = pd.merge(df, series, left_index=True, right_index=True, how='left')

        # 3. 뉴스 감성 분석
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            if news:
                sent_data = []
                for n in news:
                    pub_ts = n.get('providerPublishTime', time.time())
                    pub_date = datetime.datetime.fromtimestamp(pub_ts).date()
                    score = _self.analyzer.polarity_scores(n.get('title', ''))['compound']
                    sent_data.append({'Date': pd.Timestamp(pub_date), 'Sentiment': score})
                
                sdf = pd.DataFrame(sent_data).groupby('Date')[['Sentiment']].mean()
                sdf.index = pd.to_datetime(sdf.index).normalize()
                df = pd.merge(df, sdf, left_index=True, right_index=True, how='left')
        except: 
            pass

        # [오류 수정 포인트] AttributeError 방지 로직
        if 'Sentiment' not in df.columns:
            df['Sentiment'] = 0.0
        else:
            df['Sentiment'] = df['Sentiment'].fillna(0.0)
            
        return df.ffill().bfill()

    def calculate_indicators(self, df):
        df = df.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_High'], df['BB_Low'] = df['MA20'] + (2*std), df['MA20'] - (2*std)
        delta = df['Close'].diff()
        gain, loss = delta.where(delta > 0, 0).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        return df.fillna(50)

    def run_backtest(self, df, rsi_buy, rsi_sell):
        df = df.copy()
        df['Signal'] = 0
        df.loc[df['RSI'] < rsi_buy, 'Signal'] = 1
        df.loc[df['RSI'] > rsi_sell, 'Signal'] = -1
        df['Position'] = df['Signal'].replace(0, method='ffill').clip(lower=0)
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
        
        m_cum = (1 + df['Market_Return'].fillna(0)).cumprod().iloc[-1] - 1
        s_cum = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1
        return m_cum, s_cum

    def save_to_sheets(self, data_dict):
        """구글 시트 저장 로직 (2행 삽입 로직 유지)"""
        if not HAS_GSPREAD: return False, "라이브러리가 없습니다."
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
            client = gspread.authorize(creds)
            sheet = client.open("Wonju_Quant_Logs").sheet1
            row = [str(datetime.datetime.now())] + list(data_dict.values())
            sheet.insert_row(row, 2)
            return True, "성공적으로 저장되었습니다."
        except Exception as e:
            return False, str(e)

    def generate_gems_pack(self, df, ticker, m_ret, s_ret):
        """[Elite] 고품질 Gems 데이터 팩 생성 (국문 구조)"""
        last = df.iloc[-1]
        price_trend = "상승" if df['Close'].iloc[-1] > df['Close'].iloc[-10] else "하락"
        rsi_trend = "상승" if df['RSI'].iloc[-1] > df['RSI'].iloc[-10] else "하락"
        divergence = "발생 가능성 있음" if price_trend != rsi_trend else "없음"

        report = f"""
### 💎 원주 퀀트 연구소: Elite Analysis Data Pack ({ticker})
**분석 시각:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

#### 1. 포지션 핵심 요약 (Technical Context)
- **현재가:** ${last['Close']:.2f} | **RSI:** {last['RSI']:.2f}
- **전략 수익률(3y):** {s_ret*100:.2f}% (시장 대비: {(s_ret-m_ret)*100:+.2f}%)
- **볼린저 위치:** {'상단돌파' if last['Close']>last['BB_High'] else '하단돌파' if last['Close']<last['BB_Low'] else '정상범위'}
- **추세 괴리(Divergence):** {divergence} (주가 {price_trend} / RSI {rsi_trend})

#### 2. 매크로 및 외부 심리 (Global & News)
- **뉴스 감성(Sent):** {last['Sentiment']:.3f} (범위: -1.0 ~ 1.0)
- **변동성(VIX):** {last.get('VIX', 0):.2f} | **10Y 금리:** {last.get('US_10Y', 0):.2f}%
- **환율(USD/KRW):** {last.get('USD_KRW', 0):.2f}

#### 3. 원시 데이터 팩 (최근 5일)
{df[['Close', 'RSI', 'Sentiment', 'VIX']].tail(5).to_string()}

---
**Gems 분석 특화 프롬프트:**
"당신은 월가 출신의 퀀트 분석가입니다. 위 데이터 팩을 바탕으로 RSI-주가 간의 괴리 여부를 정밀 판독하고, VIX 수치에 기반한 현재 시장의 공포 단계를 정의하세요. 최종적으로 다음 거래일의 매수/매도 시나리오를 확률 기반으로 제안하십시오."
"""
        return report

    def plot_dashboard(self, df, ticker, rsi_buy, rsi_sell):
        """가시성 개선 차트 (White Theme 유지)"""
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.06, 
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f"{ticker} 주가 및 볼린저 밴드", "거래량", f"RSI 지표 (매수 < {rsi_buy}, 매도 > {rsi_sell})", "감성 및 VIX 지수")
        )

        # 1. Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='black', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20", line=dict(color='orange', width=1.2)), row=1, col=1)

        # 2. Volume
        colors = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple', width=1.5)), row=3, col=1)
        fig.add_hline(y=rsi_sell, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=rsi_buy, line_dash="dash", line_color="green", row=3, col=1)

        # 4. Sentiment & VIX
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color='blue', opacity=0.4), row=4, col=1)
        if 'VIX' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX", line=dict(color='red', width=1), yaxis='y2'), row=4, col=1)

        fig.update_layout(height=1000, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [UI 실행]
st.title("💎 원주 AI 퀀트 연구소 (v6.10)")

# 사이드바
with st.sidebar:
    st.header("⚙️ 제어 및 가이드")
    st.markdown("""
    <div class="gems-guide">
    <strong>💡 Gems 활용 가이드</strong><br>
    1. 분석 실행 후 하단의 데이터 팩 복사<br>
    2. Gems(ChatGPT/Claude)에 붙여넣기<br>
    3. AI가 제안하는 시나리오 검토
    </div>
    """, unsafe_allow_html=True)
    
    ticker = st.text_input("티커 (예: NVDA)", "TSLA").upper()
    period = st.selectbox("분석 기간", ["1y", "3y", "5y"], index=1)
    
    st.markdown("---")
    st.subheader("🛠️ 백테스트 설정")
    rsi_buy = st.slider("RSI 매수 기준 (과매도)", 10, 40, 30)
    rsi_sell = st.slider("RSI 매도 기준 (과매수)", 60, 90, 70)

engine = QuantLabEngine()

if st.button("🚀 전체 분석 및 동기화 실행", type="primary"):
    with st.spinner("퀀트 엔진 가동 중..."):
        df = engine.fetch_market_data(ticker, period)
        if df is not None and not df.empty:
            df = engine.calculate_indicators(df)
            m_ret, s_ret = engine.run_backtest(df, rsi_buy, rsi_sell)
            
            # KPI
            last = df.iloc[-1]
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("현재가", f"${last['Close']:.2f}", f"{(last['Close']/df.iloc[-2]['Close']-1)*100:.1f}%")
            k2.metric("RSI 전략 수익률", f"{s_ret*100:.1f}%", f"시장대비 {(s_ret-m_ret)*100:+.1f}%")
            k3.metric("감성 점수", f"{last['Sentiment']:.2f}")
            k4.metric("원/달러", f"₩{last.get('USD_KRW', 0):,.0f}")
            k5.metric("공포(VIX)", f"{last.get('VIX', 0):.2f}")
            
            # 차트
            engine.plot_dashboard(df, ticker, rsi_buy, rsi_sell)
            
            # Gems Pack & Cloud Sync
            st.markdown("---")
            st.subheader("📦 Gems 데이터 팩 & 클라우드")
            c1, c2 = st.columns([3, 1])
            with c1:
                pack_content = engine.generate_gems_pack(df, ticker, m_ret, s_ret)
                st.text_area("LLM 전송용 컨텍스트 (Elite):", pack_content, height=280)
            
            with c2:
                if st.button("💾 구글 시트 저장"):
                    log_data = {
                        "Ticker": ticker, "Price": last['Close'], "RSI": last['RSI'],
                        "Strategy_Ret": f"{s_ret*100:.2f}%", "VIX": last.get('VIX', 0)
                    }
                    if HAS_GSPREAD and "gcp_service_account" in st.secrets:
                        success, msg = engine.save_to_sheets(log_data)
                        if success: st.success(msg)
                        else: st.error(f"저장 실패: {msg}")
                    else:
                        st.warning("인증 정보(Secrets)를 확인하세요.")
                st.info("시트 저장 시 최신 데이터가 상단(2행)에 기록됩니다.")
        else:
            st.error("데이터 수집 실패. 티커를 확인해 주세요.")
