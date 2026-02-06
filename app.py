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
import time

# 1. 초기 설정 (버전 v5.1 Final: 뉴스 재시도 로직 및 안정성 강화)
st.set_page_config(page_title="Wonju AI Quant Lab v5.1", layout="wide", page_icon="💎")

# [Engineering Standard] 가용 모델 리스트 및 최적 모델 검색 함수
def get_available_ai_models():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priority = [
            'models/gemini-2.0-pro-exp', 
            'models/gemini-2.0-flash-exp',
            'models/gemini-1.5-pro', 
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]
        sorted_models = [p for p in priority if p in models]
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

# [Engineering Standard] 뉴스 가져오기 로직 보완 (v5.1 Final: 재시도 로직 추가)
def get_robust_news(ticker):
    # 최대 2번 재시도하여 일시적 네트워크 오류 극복
    max_retries = 2
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            # 첫 시도가 아니면 잠시 대기 후 요청
            if attempt > 0:
                time.sleep(1)
            
            news_data = stock.news
            
            if news_data:
                return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
            
        except Exception as e:
            # 마지막 시도에서도 실패하면 에러 메시지 반환
            if attempt == max_retries - 1:
                return f"[시스템 오류] 뉴스 데이터 수신 실패 ({str(e)})"
            continue
            
    return "[데이터 없음] 현재 야후 파이낸스에 등록된 최신 뉴스가 없습니다."

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
    
    st.subheader("🤖 AI 모델 설정")
    def format_model_name(option):
        name = option.lower()
        clean_name = option.replace('models/', '')
        if 'pro' in name: return f'🧠 Premium ({clean_name})'
        if 'flash' in name: return f'⚡ Flash ({clean_name})'
        if 'lite' in name: return f'🍃 Lite ({clean_name})'
        return clean_name

    selected_model_name = st.selectbox(
        "사용할 분석 엔진 (Brain)",
        options=available_models,
        format_func=format_model_name,
        help="Premium은 복잡한 추론에 강하고, Flash는 속도가 빠릅니다."
    )
    
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
    info_data = get_stock_info(target_ticker)
    
    current_price = last['Close']
    if len(df) >= 2:
        prev_price = df.iloc[-2]['Close']
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100
    else:
        price_change = 0
        pct_change = 0

    st.title(f"📈 {target_ticker} Pro Dashboard v5.1 Final")
    
    st.markdown("### 💰 현재 주가")
    st.metric(
        label="Price",
        value=f"{current_price:,.0f}",
        delta=f"{price_change:,.0f} ({pct_change:.2f}%)"
    )
    st.divider()
    
    display_fundamental_metrics(target_ticker)

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
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#333", font=dict(color="white")
        ),
        rangeslider_visible=False
    )
    fig.update_layout(height=800, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 3. AI 분석 섹션
    st.divider()
    
    # [수정] Gems 전용 데이터 팩 (v5.1: 뉴스 누락 시 가이드 자동 추가)
    st.subheader("🚀 Deep Research 연동 (Gems)")
    with st.expander("🔍 Gems 심층 분석용 데이터 팩 추출 (클릭하여 열기)", expanded=False):
        st.write("아래 내용을 복사하여 Gems에 붙여넣으세요. 뉴스 수집 실패 시 자동으로 보완 가이드가 포함됩니다.")
        
        news_headlines = get_robust_news(target_ticker)
        
        # 뉴스에 오류가 포함되어 있는지 체크
        news_instruction = ""
        if "데이터 없음" in news_headlines or "시스템 오류" in news_headlines:
            news_instruction = f"⚠️ 현재 실시간 뉴스 수집에 실패했습니다. 분석 전 반드시 구글 검색으로 '{target_ticker} 최신 뉴스'와 '반도체 업황'을 검색하여 팩트를 보완한 뒤 분석을 시작해줘."

        gems_data_pack = f"""
[원주 퀀트 연구소 - 실시간 데이터 팩: {target_ticker}]
- 기준일: {datetime.datetime.now().strftime('%Y-%m-%d')}
- 현재가: {current_price:,.0f} ({pct_change:.2f}%)
- 펀더멘털: PER {info_data.get('trailingPE', 'N/A')}, PBR {info_data.get('priceToBook', 'N/A')}, 배당 {info_data.get('dividendYield', 0)*100:.2f}%
- 기술적 지표: RSI(14) {last['RSI']:.1f}, 볼린저밴드 상단 {last['Upper']:,.0f} / 하단 {last['Lower']:,.0f}
- 대시보드 뉴스 요약:
{news_headlines}

{news_instruction}

[질문 가이드]
위 실시간 데이터를 기반으로, '구글 검색'을 활용하여 다음 사항을 심층 분석해줘:
1. 매크로 연결: 현재 매크로 상황(금리, 환율)이 이 종목의 기술적 과매수/과매도 상태와 어떻게 충돌하거나 공명하고 있는가?
2. 섹터 분석: 경쟁사 대비 펀더멘털 지표의 우위 점검 및 향후 1분기 예상되는 섹터 내 점유율 변화 리스크.
3. 시나리오: 위 뉴스들이 일시적 노이즈인지, 장기적 펀더멘털 훼손인지 논리적으로 비판해줘.
        """
        st.code(gems_data_pack, language="markdown")
        st.info("💡 위 텍스트를 복사한 뒤 Gems에 접속하여 '구글 검색' 기능을 켜고 질문하세요.")

    st.divider()
    
    # 대시보드 내장 빠른 분석
    display_name = format_model_name(selected_model_name)
    st.subheader(f"📢 대시보드 내장 분석 (Engine: {display_name})")
    
    if st.button("🤖 실시간 기술적 전략 분석", type="primary", use_container_width=True):
        with st.spinner(f"{display_name} 엔진 분석 중..."):
            news_headlines = get_robust_news(target_ticker)
            active_model = genai.GenerativeModel(selected_model_name)
            gen_config = {"temperature": 0.0}
            
            sentiment_prompt = f"Analyze sentiment for {target_ticker}. Headlines: {news_headlines}. Return JSON: {{'score': 0-100, 'reason': '...'}}"
            try:
                res = active_model.generate_content(sentiment_prompt, generation_config=gen_config)
                clean_json = res.text.replace('```json', '').replace('```', '')
                data = json.loads(clean_json)
                score = data.get('score', 50)
                
                col_g, col_t = st.columns([1, 2])
                with col_g: st.plotly_chart(create_sentiment_gauge(score), use_container_width=True)
                with col_t: st.info(f"{data.get('reason')} (점수: {score})")

                final_prompt = f"당신은 퀀트입니다. 가격 {last['Close']}, RSI {last['RSI']:.1f}, 뉴스점수 {score}를 근거로 [매수/관망/매도] 의견을 3줄 요약하세요."
                final_res = active_model.generate_content(final_prompt, generation_config=gen_config)
                st.success(final_res.text)
                st.toast(f"✅ {target_ticker} 분석 완료!", icon="🎉")
            except Exception as e:
                st.error(f"분석 오류: {e}")

    # 구글 시트 저장
    with st.expander("💾 투자 기록 저장"):
        if st.button("구글 시트에 현재 상태 저장"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if save_to_google_sheet(sheet_url, [now, target_ticker, float(last['Close']), float(last['RSI'])]):
                st.success("저장 완료!")
            else:
                st.error("저장 실패")
