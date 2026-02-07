import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import time

# 1. 초기 설정 (버전 v5.7: AI 내장 기능 제거 및 시트 저장 디버깅 강화)
st.set_page_config(page_title="Wonju AI Quant Lab v5.7", layout="wide", page_icon="💎")

# 2. 데이터 캐싱 및 초기화
@st.cache_data(show_spinner=False, ttl=3600)
def get_stock_info(symbol):
    # 재무 정보 수집 재시도 로직
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tick = yf.Ticker(symbol)
            info = tick.info
            if info and 'symbol' in info:
                return info
        except Exception:
            time.sleep(1)
            continue
    return None

# 3. 펀더멘털 지표 시각화
def display_fundamental_metrics(info):
    if not info:
        st.warning("⚠️ 기업 재무 정보를 불러올 수 없습니다. (차트 및 기술적 분석은 가능)")
        return

    currency = info.get('currency', 'KRW')
    market_cap = info.get('marketCap', 0)
    
    if currency == 'KRW':
        cap_display = f"{market_cap / 1_000_000_000_000:.2f}조 원"
    elif currency == 'USD':
        cap_display = f"${market_cap / 1_000_000_000:.2f} B"
    else:
        cap_display = f"{market_cap:,.0f} {currency}"

    st.markdown(f"### 🏢 {info.get('shortName', 'Unknown')} 펀더멘털(기초체력) 분석")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.metric("시가총액", cap_display)
    with col2: st.metric("PER (주가수익비율)", f"{info.get('trailingPE', 0):.2f}배" if info.get('trailingPE') else "N/A")
    with col3: st.metric("PBR (주가순자산비율)", f"{info.get('priceToBook', 0):.2f}배" if info.get('priceToBook') else "N/A")
    with col4: st.metric("배당수익률", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
    st.divider()

# 4. 구글 시트 저장 (디버깅 모드 적용)
def save_to_google_sheet(url, data):
    try:
        # Streamlit Secrets 확인
        if "gcp_service_account" not in st.secrets:
            st.error("❌ 설정 오류: 'secrets.toml' 파일에 구글 인증 정보(gcp_service_account)가 없습니다.")
            return False

        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True
    except Exception as e:
        # 에러 발생 시 구체적인 이유 출력
        st.error(f"❌ 저장 실패: {str(e)}")
        return False

# 5. 뉴스 가져오기
def get_robust_news(ticker):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            if attempt > 0: time.sleep(1)
            news_data = stock.news
            
            if isinstance(news_data, list) and len(news_data) > 0:
                news_list = []
                for n in news_data[:5]:
                    if isinstance(n, dict):
                        title = n.get('title', '제목 정보 없음')
                        publisher = n.get('publisher', '출처 미상')
                        news_list.append(f"- {title} ({publisher})")
                
                if news_list:
                    return "\n".join(news_list)
            
            return "[데이터 없음] 현재 야후 파이낸스에 등록된 뉴스가 없습니다."
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[시스템 오류] 뉴스 수신 일시 장애 (사유: {str(e)})"
            continue
    return "[데이터 없음] 최신 뉴스가 없습니다."

# 6. 기술적 데이터 계산
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
    st.caption("v5.7 Lite & Stable")
    
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
    info_data = get_stock_info(target_ticker) or {}
    
    current_price = last['Close']
    if len(df) >= 2:
        prev_price = df.iloc[-2]['Close']
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100
    else:
        price_change = 0
        pct_change = 0

    st.title(f"📈 {target_ticker} Pro Dashboard")
    
    st.markdown("### 💰 현재 주가")
    st.metric(
        label="Price",
        value=f"{current_price:,.0f}",
        delta=f"{price_change:,.0f} ({pct_change:.2f}%)"
    )
    st.divider()
    
    display_fundamental_metrics(info_data)

    st.subheader("📊 기술적 분석 차트")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="상단", line=dict(dash='dot', color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="하단", line=dict(dash='dot', color='white')), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=3, col=1)
    
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

    # [v5.7] Gems 연동 섹션 (심플하게 유지)
    st.divider()
    st.subheader("🚀 Deep Research 데이터 팩")
    with st.expander("✅ Gems 심층 분석용 데이터 팩 추출", expanded=True):
        news_headlines = get_robust_news(target_ticker)
        sector = info_data.get('sector', 'Unknown')
        
        sector_guidance = {
            "Technology": "반도체 사이클 및 기술 격차 중점 점검.",
            "Financial Services": "금리 사이클 및 주주 환원 정책 점검.",
            "Consumer Defensive": "원자재 가격 변동성 및 내수 소비 트렌드 점검."
        }.get(sector, "업계 경쟁력 및 시장 점유율 점검.")

        # [중요] 뉴스 오류 시 자동 가이드 생성 로직
        news_instruction = ""
        if "데이터 없음" in news_headlines or "시스템 오류" in news_headlines:
            news_instruction = f"\n⚠️ [주의] 뉴스 데이터 수집이 원활하지 않습니다. 구글 검색으로 '{target_ticker} 최신 이슈'를 반드시 직접 검색하여 분석에 반영하세요.\n"

        master_prompt = f"""
[원주 퀀트 연구소 - 실시간 데이터 팩: {target_ticker}]
- 기준일: {datetime.datetime.now().strftime('%Y-%m-%d')}
- 현재가: {current_price:,.0f} ({pct_change:.2f}%)
- 펀더멘털: PER {info_data.get('trailingPE', 'N/A')}, PBR {info_data.get('priceToBook', 'N/A')}
- 섹터: {sector}
- 기술적 상태: RSI(14) {last['RSI']:.1f}, 볼린저밴드 하단 {last['Lower']:,.0f}
- 대시보드 뉴스 요약:
{news_headlines}
{news_instruction}
---
[심층 분석 지침]
1. 데이터 그라운딩: 지표와 뉴스 간 괴리 분석.
2. 섹터 특화 분석 ({sector}): {sector_guidance}
3. 악마의 변호인: 매수 논리를 무력화할 리스크 2가지를 찾으세요.
4. 최종 결론: [매수/관망/매도] 중 선택하고, 특히 [손절가]를 명확히 제시하세요.
        """
        st.code(master_prompt, language="markdown")
        st.info("💡 위 텍스트를 복사하여 제미나이에 붙여넣으세요.")

    # 구글 시트 저장
    with st.expander("💾 투자 기록 저장"):
        if st.button("구글 시트에 현재 상태 저장"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if save_to_google_sheet(sheet_url, [now, target_ticker, float(last['Close']), float(last['RSI'])]):
                st.success("저장 완료!")
            # 실패 시 에러 메시지는 save_to_google_sheet 함수 내부에서 출력됨

    st.divider()
    st.caption("💎 원주 퀀트 연구소 v5.7 - 데이터 기반 의사결정 시스템")
