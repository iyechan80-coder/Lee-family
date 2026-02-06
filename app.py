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

# 1. 초기 설정 (버전 v4.6: 모델 선택 기능 추가)
st.set_page_config(page_title="Wonju AI Quant Lab v4.6", layout="wide", page_icon="💎")

# [Engineering Standard] 가용 모델 리스트 및 최적 모델 검색 함수
def get_available_ai_models():
    try:
        # generateContent를 지원하는 모델 리스트 확보
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 선호도 순서대로 정렬 (Pro > Flash > Legacy Pro)
        priority = ['models/gemini-1.5-pro', 'models/gemini-1.5-flash', 'models/gemini-pro']
        sorted_models = [p for p in priority if p in models]
        # 리스트에 없는 기타 모델들 추가
        remaining = [m for m in models if m not in priority]
        return sorted_models + remaining
    except Exception:
        return ['gemini-pro']

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    available_models = get_available_ai_models()
else:
    st.error("⚠️ secrets.toml에 GOOGLE_API_KEY가 없습니다.")
    available_models = []

# 2. 데이터 캐싱 및 초기화
@st.cache_data(show_spinner=False, ttl=3600)
def get_stock_info(symbol):
    try:
        tick = yf.Ticker(symbol)
        info = tick.info
        if 'symbol' not in info: return None
        return info
    except Exception:
        return None

# 3. 펀더멘털 지표 시각화
def display_fundamental_metrics(ticker_symbol):
    info = get_stock_info(ticker_symbol)
    if info is None:
        st.warning(f"⚠️ '{ticker_symbol}' 정보를 불러올 수 없습니다.")
        return

    currency = info.get('currency', 'KRW')
    market_cap = info.get('marketCap', 0)
    if currency == 'KRW':
        cap_display = f"{market_cap / 1_000_000_000_000:.2f}조 원"
    elif currency == 'USD':
        cap_display = f"${market_cap / 1_000_000_000:.2f} B"
    else:
        cap_display = f"{market_cap:,.0f} {currency}"

    st.markdown(f"### 🏢 {info.get('shortName', ticker_symbol)} 펀더멘털(기초체력) 분석")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("시가총액", cap_display)
    with col2: st.metric("PER (주가수익비율)", f"{info.get('trailingPE', 0):.2f}배" if info.get('trailingPE') else "N/A")
    with col3: st.metric("PBR (주가순자산비율)", f"{info.get('priceToBook', 0):.2f}배" if info.get('priceToBook') else "N/A")
    with col4: st.metric("배당수익률", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
    st.divider()

# 4. 구글 시트 저장
def save_to_google_sheet(url, data):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True
    except Exception:
        return False

# 5. 뉴스 가져오기
def get_robust_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data: return "최근 관련 뉴스가 없습니다."
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except Exception:
        return "뉴스를 불러오는 중 오류가 발생했습니다."

# 6. 게이지 차트
def create_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI 뉴스 감성 점수"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 40], 'color': '#ff4b4b'},
                {'range': [40, 60], 'color': '#faca2b'},
                {'range': [60, 100], 'color': '#09ab3b'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
    return fig

# 7. 데이터 계산
@st.cache_data(ttl=3600)
def get_advanced_data(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
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

# --- 메인 실행 ---
with st.sidebar:
    st.header("🔍 원주 퀀트 연구소")
    
    # [UX 개선] AI 모델 선택기 추가
    st.subheader("🤖 AI 모델 설정")
    selected_model_name = st.selectbox(
        "사용할 분석 엔진 (Brain)",
        options=available_models,
        help="Pro 모델이 가장 똑똑하지만 느릴 수 있고, Flash 모델은 빠릅니다."
    )
    
    # [디버깅] 캐시 삭제 버튼
    if st.button("🗑️ 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("기간", ["6mo", "1y", "3y"])
    sheet_url = st.text_input("구글 시트 URL", value="https://docs.google.com/spreadsheets/d/1cDwpOaZfEDJY6v7aZa92A9KgRHFqT8S7jy9jywc5rRY/edit?usp=sharing")

df = get_advanced_data(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    
    st.title(f"📈 {target_ticker} Pro Dashboard v4.6")
    
    # 1. 펀더멘털 분석
    display_fundamental_metrics(target_ticker)

    # 2. 차트 분석
    st.subheader("📊 기술적 분석 차트")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="상단", line=dict(dash='dot', color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="하단", line=dict(dash='dot', color='white')), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color="red", dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color="green", dash="dot"), row=3, col=1)
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 3. AI 분석
    st.divider()
    st.subheader(f"📢 AI 정밀 분석 (Engine: {selected_model_name.split('/')[-1]})")
    
    if st.button("🤖 뉴스 감성 + 전략 분석 실행", type="primary", use_container_width=True):
        with st.spinner(f"{selected_model_name} 엔진이 데이터를 분석 중입니다..."):
            news_headlines = get_robust_news(target_ticker)
            
            # 선택된 모델로 인스턴스 생성
            active_model = genai.GenerativeModel(selected_model_name)
            gen_config = {"temperature": 0.0}

            sentiment_prompt = f"""
            Analyze the sentiment of: {news_headlines} for {target_ticker}.
            Return JSON: {{"score": 50, "reason": "summary..."}}
            """
            
            try:
                # 1단계: 감성 분석
                res = active_model.generate_content(sentiment_prompt, generation_config=gen_config)
                clean_json = res.text.replace('```json', '').replace('```', '')
                data = json.loads(clean_json)
                score = data.get('score', 50)
                
                col_g, col_t = st.columns([1, 2])
                with col_g: st.plotly_chart(create_sentiment_gauge(score), use_container_width=True)
                with col_t: st.info(f"{data.get('reason')} (감성 점수: {score})")

                # 2단계: 종합 전략
                final_prompt = f"""
                당신은 냉철한 퀀트 트레이더입니다. {selected_model_name}의 논리력을 발휘하여 답변하세요.
                데이터: 현재가 {last['Close']}, RSI {last['RSI']:.1f}, 뉴스점수 {score}
                뉴스요약: {data.get('reason')}
                결론을 [강력 매수/분할 매수/관망/매도] 중 하나로 시작하고, 퀀트적 근거를 포함하여 3줄로 요약하세요.
                """
                final_res = active_model.generate_content(final_prompt, generation_config=gen_config)
                st.write("### 🗣️ 트레이더 의견")
                st.success(final_res.text)

            except Exception as e:
                st.error(f"분석 오류 ({selected_model_name}): {e}")

    # 4. 저장
    with st.expander("💾 투자 기록 저장"):
        if st.button("구글 시트에 현재 상태 저장"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if save_to_google_sheet(sheet_url, [now, target_ticker, float(last['Close']), float(last['RSI'])]):
                st.success("저장 완료!")
            else:
                st.error("저장 실패")
