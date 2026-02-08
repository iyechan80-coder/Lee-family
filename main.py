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

# 구글 시트 연동 라이브러리
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

# [초기 설정]
st.set_page_config(page_title="Wonju AI Quant Lab v6.30", layout="wide", page_icon="💎")

# [전역 스타일 설정]
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
    .gems-guide-main {
        background-color: #FFF5F5;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #E53E3E;
        margin: 20px 0;
    }
    .protocol-step {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 8px;
        border: 1px dashed #FC8181;
    }
    /* 코드 블록 스타일 (Copy 버튼 가시성 확보) */
    .stCodeBlock {
        border: 2px solid #3182CE !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# [내장형 금융 감성 분석기]
class LiteSentimentAnalyzer:
    def __init__(self):
        self.pos_words = {
            'up', 'rise', 'gain', 'bull', 'high', 'growth', 'profit', 'jump', 'surge', 
            'record', 'beat', 'buy', 'positive', 'good', 'outperform', 'dividend', 
            'upgrade', 'soar', 'bullish', 'guidance', 'recovery', 'expansion'
        }
        self.neg_words = {
            'down', 'fall', 'loss', 'bear', 'low', 'drop', 'crash', 'miss', 'risk', 
            'debt', 'sell', 'negative', 'concern', 'fail', 'bad', 'underperform', 
            'downgrade', 'plunge', 'bearish', 'inflation', 'recession', 'volatile',
            'overvalued', 'bubble', 'lawsuit', 'bankruptcy'
        }

    def polarity_scores(self, text):
        text = str(text).lower()
        words = re.findall(r'\w+', text)
        p_count = sum(1 for w in words if w in self.pos_words)
        n_count = sum(1 for w in words if w in self.neg_words)
        score = p_count - n_count
        norm_score = score / (p_count + n_count + 1)
        return {'compound': norm_score}

class QuantLabEngine:
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.analyzer_type = "Vader (Adv)"
        except:
            self.analyzer = LiteSentimentAnalyzer()
            self.analyzer_type = "Lite (Fin-Optimized)"

    def _clean_index(self, df):
        if df.empty: return df
        # 타임존 제거 및 정규화 (데이터 병합 충돌 방지 핵심)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
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
        df = _self._fetch_with_retry(ticker, period)
        if df is None or df.empty: return None
        df = _self._clean_index(df)

        macro_map = {"^VIX": "VIX", "^TNX": "US_10Y", "KRW=X": "USD_KRW"}
        for m_ticker, col in macro_map.items():
            m_df = _self._fetch_with_retry(m_ticker, period)
            if not m_df.empty:
                m_df = _self._clean_index(m_df)
                if 'Close' in m_df.columns:
                    series = m_df[['Close']].rename(columns={'Close': col})
                    df = pd.merge(df, series, left_index=True, right_index=True, how='left')

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
        except: pass

        if 'Sentiment' not in df.columns: df['Sentiment'] = 0.0
        else: df['Sentiment'] = df['Sentiment'].fillna(0.0)
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
        """[보완] Pandas 최신 문법 적용 (method='ffill' 제거)"""
        df = df.copy()
        df['Signal'] = 0
        df.loc[df['RSI'] < rsi_buy, 'Signal'] = 1
        df.loc[df['RSI'] > rsi_sell, 'Signal'] = -1
        
        # [수정] deprecated 경고 방지를 위한 표준 문법
        df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).clip(lower=0)
        
        df['Market_Return'] = df['Close'].pct_change().fillna(0)
        df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
        df['Strategy_Return'] = df['Strategy_Return'].fillna(0)

        m_cum = (1 + df['Market_Return']).cumprod().iloc[-1] - 1
        s_cum = (1 + df['Strategy_Return']).cumprod().iloc[-1] - 1

        cum_equity = (1 + df['Strategy_Return']).cumprod()
        running_max = cum_equity.cummax()
        drawdown = (cum_equity - running_max) / running_max
        mdd = drawdown.min()

        df['Trade'] = df['Position'].diff()
        entries = df[df['Trade'] == 1].index
        exits = df[df['Trade'] == -1].index
        
        wins = 0
        trade_count = min(len(entries), len(exits))
        
        if trade_count > 0:
            for i in range(trade_count):
                if df.loc[exits[i]]['Close'] > df.loc[entries[i]]['Close']:
                    wins += 1
            win_rate = (wins / trade_count) * 100
        else:
            win_rate = 0.0

        return m_cum, s_cum, mdd, win_rate, trade_count

    def save_to_sheets(self, data_dict):
        if not HAS_GSPREAD: return False, "라이브러리(gspread) 미설치 상태입니다."
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            if "gcp_service_account" not in st.secrets: return False, "Secrets 인증 정보가 없습니다."
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
            client = gspread.authorize(creds)
            sheet = client.open("Wonju_Quant_Logs").sheet1
            sheet.insert_row([str(datetime.datetime.now())] + list(data_dict.values()), 2)
            return True, "클라우드(2행)에 성공적으로 기록되었습니다."
        except Exception as e: return False, f"연동 에러: {str(e)}"

    def generate_gems_pack(self, df, ticker, m_ret, s_ret, mdd, win_rate, trades, horizon):
        """[Final Split] 데이터와 프롬프트 분리 생성"""
        last = df.iloc[-1]
        price_trend = "Upward" if df['Close'].iloc[-1] > df['Close'].iloc[-10] else "Downward"
        rsi_trend = "Upward" if df['RSI'].iloc[-1] > df['RSI'].iloc[-10] else "Downward"
        divergence = "Potential Divergence" if price_trend != rsi_trend else "None"

        # 1. 데이터 팩 (Data Only)
        data_pack = f"""
[Wonju Quant Lab Analysis Data Pack: {ticker}]
Investment Horizon: {horizon}
Analysis Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

#### SECTION A. PERFORMANCE METRICS (3y Backtest)
- Ticker: {ticker}
- Price: ${last['Close']:.2f}
- RSI(14): {last['RSI']:.2f}
- Strategy Return: {s_ret*100:.2f}% (Market Bench: {m_ret*100:.2f}%)
- Max Drawdown (MDD): {mdd*100:.2f}% (Risk Sensitivity Check)
- Win Rate: {win_rate:.1f}% ({trades} Trades Executed)
- Bollinger Position: {'Over Upper' if last['Close']>last['BB_High'] else 'Under Lower' if last['Close']<last['BB_Low'] else 'Neutral'}
- Trend Divergence: {divergence}

#### SECTION B. MACRO & SENTIMENT
- Fear Index (VIX): {last.get('VIX', 0):.2f}
- 10Y Bond Yield: {last.get('US_10Y', 0):.2f}%
- Exchange Rate (USD/KRW): {last.get('USD_KRW', 0):.2f}
- Sentiment Score: {last['Sentiment']:.3f}

#### SECTION C. RECENT TREND (Last 5 Days)
{df[['Close', 'RSI', 'Sentiment', 'VIX']].tail(5).to_string()}
"""

        # 2. 투자 기간별 동적 가중치 설정
        if "단기" in horizon:
            horizon_guide = "실시간 변동성, RSI 과매도/과매수, 뉴스 감성 및 수급 모멘텀"
        elif "중기" in horizon:
            horizon_guide = "이평선 추세 정배열 여부, 볼린저 밴드 이탈 방향, 분기 실적 전망"
        else:
            horizon_guide = "10년물 금리 및 환율 매크로 환경, 산업 내 독점력, 장기 밸류에이션(P/E, P/B)"

        # 3. 수석 전략가 지시사항 (Instruction Only)
        system_prompt = f"""
[Identity & Role]
당신은 '원주 퀀트 연구소'의 수석 트레이딩 전략가(Chief Strategist)입니다. 당신의 최우선 가치는 **'사용자의 원금 보호'**입니다. 감정적인 희망 회로를 철저히 배제하고, 데이터가 부정적일 경우 어설픈 대안 대신 단호한 **[매수 금지]**를 선언하십시오.

[Operational Protocol: 4단계 분석 프로세스]
Phase 1. 능동적 팩트 체크 (Google Search 필수)
- 제공된 데이터 팩의 뉴스 섹션이 부실할 경우, 반드시 '{ticker}' 티커를 기반으로 구글 검색을 수행하십시오.
- **{horizon} 분석 가중치:** {horizon_guide}에 가장 높은 비중을 두고 리서치하십시오.

Phase 2. 데이터 그라운딩 (Data Grounding)
- 기술적 지표(RSI/BB/MDD)와 리서치한 뉴스 간의 괴리를 분석하십시오. 특히 MDD {mdd*100:.1f}%가 '{horizon}' 기간 동안 투자자가 견딜 수 있는 수준인지 평가하십시오.

Phase 3. 리스크 검증 (Devil's Advocate)
- [필수] "이 종목을 지금 사면 망하는 이유 2가지"를 가장 냉정하게 제시하십시오.

Phase 4. 트레이딩 셋업 (Binary Decision)
- [BUY/PASS]: 아래 4개 조건을 모두 충족할 때만 매수 전략을 출력하십시오.
  1. 주가 추세가 {horizon} 관점에서 매수 우위임.
  2. 명확한 상승 모멘텀(뉴스/재료)이 검색됨.
  3. RSI가 과매수 상태가 아님.
  4. MDD가 리스크 관리 범위(-20% 이내 권장)에 있음.
- [AVOID/PROHIBITED]: 위 조건 중 하나라도 미달하면 즉시 **[매수 금지]**를 선언하고 진입가/목표가를 삭제하십시오.

[Output Format]
📊 '{horizon}' 심층 분석 요약
🛡️ 리스크 점검 (매수 금지 사유 최상단 배치)
🎯 트레이딩 전략 (Action Plan)
- 판단: [강력 매수 / 관망 / 매수 금지] 중 택 1
- 전략: '{horizon}'에 맞는 진입가 및 목표가 제시 (매수 금지 시 삭제)
- ⛔ 손절가 (Stop-loss): 매수 전략일 경우 필수 작성.

👨‍👩‍👧‍👦 가족을 위한 한 줄 브리핑
예: "상한 사과입니다. 겉이 번지르르해도 절대 한 입 베어 물지 마세요."

[System Rules]
- 단호함: 모호한 추측 대신 데이터에 기반한 결론만 내리십시오.
- 데이터 태그: 답변 최하단에 ###DATA_START### [판단] 핵심 근거 한 줄 요약 ###DATA_END###를 포함하십시오.
"""
        return data_pack, system_prompt

    def plot_dashboard(self, df, ticker, rsi_buy, rsi_sell):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.15, 0.15, 0.2],
                           subplot_titles=(f"{ticker} 주가 및 볼린저 밴드", "거래량", f"RSI 지표 (Buy < {rsi_buy}, Sell > {rsi_sell})", "감성 및 VIX 지수"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='black', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dot', color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dot', color='gray'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA 20", line=dict(color='orange', width=1.2)), row=1, col=1)
        colors = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple', width=1.5)), row=3, col=1)
        fig.add_hline(y=rsi_sell, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=rsi_buy, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Sentiment'], name="Sentiment", marker_color='blue', opacity=0.4), row=4, col=1)
        if 'VIX' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX", line=dict(color='red', width=1), yaxis='y2'), row=4, col=1)
        fig.update_layout(height=1000, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

# [UI 실행]
st.title("💎 원주 AI 퀀트 연구소 (v6.30)")

with st.sidebar:
    st.header("⚙️ 제어 패널")
    ticker = st.text_input("티커 (예: TSLA)", "TSLA").upper()
    period = st.selectbox("분석 기간", ["1y", "3y", "5y"], index=1)
    
    st.markdown("---")
    st.subheader("🎯 투자 호라이즌 설정")
    horizon = st.radio("투자 목표 기간", [
        "단기 (1~14일)", 
        "중기 (2주~6개월)", 
        "장기 (6개월 이상)"
    ])
    
    st.markdown("---")
    st.subheader("🛠️ 백테스트 설정 (실시간)")
    rsi_buy = st.slider("RSI 매수 기준 (과매도)", 10, 40, 30, key='rsi_buy_slider')
    rsi_sell = st.slider("RSI 매도 기준 (과매수)", 60, 90, 70, key='rsi_sell_slider')

engine = QuantLabEngine()
if 'analyzed_data' not in st.session_state: st.session_state.analyzed_data = None

# 1. 데이터 수집 버튼
if st.button("🚀 전체 분석 실행", type="primary"):
    with st.spinner("수석 전략가 엔진 가동 중..."):
        df = engine.fetch_market_data(ticker, period)
        if df is not None and not df.empty:
            df = engine.calculate_indicators(df)
            # 호라이즌 정보도 세션에 저장
            st.session_state.analyzed_data = {'df': df, 'ticker': ticker, 'horizon': horizon}
        else: st.error("데이터 수집 실패. 티커를 확인해 주세요.")

# 2. 결과 렌더링 및 동적 백테스트
if st.session_state.analyzed_data:
    res = st.session_state.analyzed_data
    df, t_name = res['df'], res['ticker']
    # 저장된 호라이즌이 없으면 현재 선택값 사용 (실시간 변경 대비)
    saved_horizon = res.get('horizon', horizon)
    
    # 동적 재계산
    m_ret, s_ret, mdd, win_rate, total_trades = engine.run_backtest(df, rsi_buy, rsi_sell)
    last = df.iloc[-1]
    
    # KPI
    st.markdown(f"### 📊 Key Performance Indicators ({saved_horizon} 관점)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("현재가", f"${last['Close']:.2f}", f"{(last['Close']/df.iloc[-2]['Close']-1)*100:.1f}%")
    k2.metric("전략 수익률", f"{s_ret*100:.1f}%", f"존버(Buy&Hold) {m_ret*100:.1f}%")
    k3.metric("최대 낙폭 (MDD)", f"{mdd*100:.2f}%", "Risk Check", delta_color="inverse")
    k4.metric("승률 (Win Rate)", f"{win_rate:.1f}%", f"{total_trades}회 매매")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("뉴스 감성", f"{last['Sentiment']:.2f}")
    k6.metric("원/달러", f"₩{last.get('USD_KRW', 0):,.0f}")
    k7.metric("공포(VIX)", f"{last.get('VIX', 0):.2f}")
    k8.metric("미국채 10년", f"{last.get('US_10Y', 0):.2f}%")
    
    engine.plot_dashboard(df, t_name, rsi_buy, rsi_sell)
    
    st.markdown("""
        <div class="gems-guide-main">
            <h2 style='color: #E53E3E;'>🛡️ 수석 트레이딩 전략가 분석 프로토콜</h2>
            <p>본 데이터 팩은 <b>원금 보호</b>를 최우선으로 분석하도록 설계되었습니다. 주변 동료들과 공유 시 아래 단계를 반드시 준수하십시오.</p>
            <div class="protocol-step"><b>Step 1.</b> 아래 두 개의 박스(데이터, 프롬프트) 우측 상단 <b>📄(복사)</b> 버튼을 각각 누릅니다.</div>
            <div class="protocol-step"><b>Step 2.</b> Gems(ChatGPT/Claude)에 순서대로 붙여넣습니다.</div>
            <div class="protocol-step"><b>Step 3.</b> AI가 제시한 <b>분석 결과</b>를 정독한 뒤 최종 의사결정을 내립니다.</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📦 Gems 데이터 팩 & 클라우드 동기화")
    c1, c2 = st.columns([3, 1])
    with c1:
        # horizon 정보를 포함하여 데이터 팩 생성
        data_pack, system_prompt = engine.generate_gems_pack(df, t_name, m_ret, s_ret, mdd, win_rate, total_trades, saved_horizon)
        
        st.caption(f"1️⃣ 데이터 팩 (Horizon: {saved_horizon})")
        st.code(data_pack, language="yaml")
        
        st.caption("2️⃣ 수석 전략가 지시사항 (System Prompt)")
        st.code(system_prompt, language="yaml")
        
    with c2:
        if st.button("💾 구글 시트 저장"):
            log_data = {"Ticker": t_name, "Price": last['Close'], "RSI": last['RSI'], "Strategy_Ret": f"{s_ret*100:.2f}%", "MDD": f"{mdd*100:.2f}%", "Win_Rate": f"{win_rate:.1f}%", "Horizon": saved_horizon}
            success, msg = engine.save_to_sheets(log_data)
            if success: st.success(msg)
            else: st.error(msg)
        st.info("저장 시 최신 분석 결과와 '투자 기간'이 시트 상단(2행)에 자동 기록됩니다.")
