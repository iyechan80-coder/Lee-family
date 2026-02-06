import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
import json

# 1. 초기 설정 및 보안 연결
st.set_page_config(page_title="Wonju AI Quant Lab Pro v4.2", layout="wide", page_icon="🔥")

# 모델 로드 (안전 장치 포함)
def get_stable_model():
    try:
        # 사용 가능한 모델 리스트 확인
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = 'models/gemini-1.5-flash'
        # 리스트에 해당 모델이 있으면 사용, 없으면 첫 번째 가용 모델 선택
        return genai.GenerativeModel(target if target in available_models else available_models[0])
    except Exception:
        # API 호출 실패 시 기본 모델로 폴백
        return genai.GenerativeModel('gemini-pro')

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = get_stable_model()
else:
    st.error("⚠️ secrets.toml에 GOOGLE_API_KEY가 없습니다.")

# 2. 펀더멘털 데이터 수집 및 캐싱
@st.cache_data(show_spinner=False, ttl=3600)
def get_stock_info(symbol):
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        if 'symbol' not in info: return None
        return info
    except Exception:
        return None

# 3. 펀더멘털 지표 시각화 함수
def display_fundamental_metrics(ticker_symbol):
    info = get_stock_info(ticker_symbol)
    
    if info is None:
        st.warning(f"⚠️ '{ticker_symbol}' 정보를 불러올 수 없습니다.")
        return

    # 화폐 단위 및 포맷 자동화
    currency = info.get('currency', 'KRW')
    market_cap = info.get('marketCap', 0)
    
    if currency == 'KRW':
        cap_display = f"{market_cap / 1_000_000_000_000:.2f}조 원"
    elif currency == 'USD':
        cap_display = f"${market_cap / 1_000_000_000:.2f} B"
    else:
        cap_display = f"{market_cap:,.0f} {currency}"

    # UI 레이아웃
    st.markdown(f"### 🏢 {info.get('shortName', ticker_symbol)} 펀더멘털 개요")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("시가총액", cap_display)
    with col2:
        per = info.get('trailingPE')
        st.metric("PER (주가수익비율)", f"{per:.2f}배" if per else "N/A")
    with col3:
        pbr = info.get('priceToBook')
        st.metric("PBR (주가순자산비율)", f"{pbr:.2f}배" if pbr else "N/A")
    with col4:
        div = info.get('dividendYield')
        st.metric("배당수익률", f"{div*100:.2f}%" if div else "N/A")

    website = info.get('website', '#')
    st.caption(
        f"📌 **섹터**: {info.get('sector', '-')} | "
        f"**산업**: {info.get('industry', '-')} | "
        f"[홈페이지]({website})"
    )
    st.divider()

# 4. 구글 시트 저장 함수
def save_to_google_sheet(url, data):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True
    except Exception as e:
        st.error(f"시트 저장 실패: {e}")
        return False

# 5. 뉴스 수집 함수
def get_robust_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data: return "최근 관련 뉴스가 없습니다."
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except Exception:
        return "뉴스를 불러오는 중 오류가 발생했습니다."

# [NEW] 6. 감성 분석 게이지 차트 생성 함수
def create_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI 뉴스 감성 점수"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # 투명 바 (화살표 대신 색상 구간으로 표시)
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ff4b4b'},  # 부정 (Red)
                {'range': [40, 60], 'color': '#faca2b'}, # 중립 (Yellow)
                {'range': [60, 100], 'color': '#09ab3b'} # 긍정 (Green)
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

# 7. 테크니컬 데이터 계산
@st.cache_data(ttl=3600)
def get_advanced_data(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        
        # 보조지표 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (std * 2)
        df['Lower'] = df['MA20'] - (std * 2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        return df
    except Exception:
        return None

# --- 메인 UI 구성 ---
with st.sidebar:
    st.header("🔍 원주 퀀트 연구소")
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("분석 기간", ["6mo", "1y", "3y"], index=0)
    
    sheet_url = st.text_input(
        "구글 시트 URL", 
        value="https://docs.google.com/spreadsheets/d/1cDwpOaZfEDJY6v7aZa92A9KgRHFqT8S7jy9jywc5rRY/edit?usp=sharing"
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: 한국 주식은 '.KS', 미국 주식은 티커만 입력하세요.")

# 메인 로직 실행
df = get_advanced_data(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🔥 {target_ticker} 딥 다이브 대시보드")
    
    # [통합] 1. 펀더멘털 분석
    display_fundamental_metrics(target_ticker)

    # [통합] 2. 테크니컬 차트
    st.subheader("📈 기술적 차트 분석")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    
    # 캔들차트 & 볼린저밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="BB 상단", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="BB 하단", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)

    # 거래량
    colors = ['red' if row['Open'] < row['Close'] else 'blue' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량", marker_color=colors), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI (14)", line=dict(color='orange')), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color="green", width=1, dash="dot"), row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # [통합] 3. AI 감성 분석 & 전략 리포트 (Phase 2 업그레이드)
    st.divider()
    st.subheader("📢 AI 감성 & 전략 리포트")
    st.caption("👇 아래 버튼을 누르면 최신 뉴스를 분석하여 호재/악재 점수와 투자 전략을 보여줍니다.")

    # 버튼을 누르면 AI 분석 시작
    if st.button("🤖 뉴스 감성 + 퀀트 전략 분석 실행", type="primary", use_container_width=True):
        with st.spinner("AI가 뉴스 감성을 채점하고 차트를 분석 중입니다..."):
            news_headlines = get_robust_news(target_ticker)
            
            # [Step 1] 뉴스 감성 점수 산출 (JSON 포맷 요청)
            sentiment_prompt = f"""
            Analyze the sentiment of the following news headlines for {target_ticker}.
            Return ONLY a JSON object with a 'score' (0-100, where 0 is very negative, 100 is very positive) and a short 'reason'.
            Headlines: {news_headlines}
            Format: {{"score": 50, "reason": "summary..."}}
            """
            
            try:
                # 감성 점수 추출
                sentiment_res = model.generate_content(sentiment_prompt)
                # JSON 파싱 (혹시 모를 마크다운 태그 제거)
                sentiment_text = sentiment_res.text.replace('```json', '').replace('```', '')
                sentiment_data = json.loads(sentiment_text)
                score = sentiment_data.get('score', 50)
                reason = sentiment_data.get('reason', '뉴스 분석 불가')
                
                # [Step 2] 결과 시각화 (좌: 게이지 차트 / 우: 텍스트 요약)
                col_gauge, col_text = st.columns([1, 2])
                with col_gauge:
                    st.plotly_chart(create_sentiment_gauge(score), use_container_width=True)
                
                with col_text:
                    st.markdown(f"**📰 뉴스 요약 및 감성 분석**")
                    st.info(f"{reason} (점수: {score}/100)")
                    
                # [Step 3] 최종 매매 전략 수립 (점수 + 차트 데이터 결합)
                final_prompt = f"""
                당신은 원주 퀀트 연구소 수석 트레이더입니다.
                [데이터] 현재가: {last['Close']:,.0f}, RSI: {last['RSI']:.1f}, 뉴스 감성점수: {score}/100
                [뉴스 요약] {reason}
                위 데이터를 종합하여 [적극 매수/분할 매수/관망/매도] 중 하나의 결론을 내리고, 가족들이 이해하기 쉽게 3줄로 요약해 주세요.
                """
                final_res = model.generate_content(final_prompt)
                st.markdown("### 🗣️ 수석 트레이더의 조언")
                st.write(final_res.text)

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                st.write("상세 에러:", e)

    st.divider()
    # 저장 기능 하단 배치 (UI 정리)
    with st.expander("💾 데이터 기록 열기"):
        st.caption("현재 주가와 RSI 상태를 구글 시트에 저장합니다.")
        if st.button("🚀 투자 기록 저장"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            log_data = [now, target_ticker, float(last['Close']), float(last['RSI'])]
            if save_to_google_sheet(sheet_url, log_data):
                st.toast("✅ 구글 시트에 저장되었습니다!", icon="📝")
            else:
                st.error("저장 실패 (URL 권한을 확인하세요)")

else:
    st.warning("⚠️ 종목 정보를 불러올 수 없습니다. 올바른 티커인지 확인해주세요.")
