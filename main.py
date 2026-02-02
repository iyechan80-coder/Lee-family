import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="우리 가족 주식 분석기", layout="wide")
st.title("📊 월가 퀀트 스타일 주식 분석 대시보드")
st.markdown("가족들을 위해 제작된 실시간 주식 분석 도구입니다. 왼쪽 사이드바에서 종목을 변경해보세요.")

# 2. 사이드바 설정 (가족들이 조작할 부분)
st.sidebar.header("🔍 분석 설정")
target_ticker = st.sidebar.text_input("종목 코드를 입력하세요 (예: NVDA, 005930.KS)", value="005930.KS")
period_choice = st.sidebar.selectbox("조회 기간 선택", ["1y", "6mo", "2y", "3y"])

def analyze_ultimate_st(ticker):
    try:
        # 데이터 수집
        stock = yf.Ticker(ticker)
        df = stock.history(period=period_choice)
        
        if df.empty:
            st.error(f"❌ '{ticker}' 데이터를 찾을 수 없습니다. 코드를 확인해주세요.")
            return

        # 핵심 지표 계산 (기존 로직 유지)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (std_dev * 2)
        df['Lower_Band'] = df['MA20'] - (std_dev * 2)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df = df.dropna()

        # 차트 시각화
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # [메인] 주가 + 볼린저 밴드
        ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
        ax1.plot(df.index, df['MA20'], label='MA20', color='orange', linestyle='--')
        ax1.plot(df.index, df['Upper_Band'], label='Upper Band', color='red', alpha=0.3)
        ax1.plot(df.index, df['Lower_Band'], label='Lower Band', color='blue', alpha=0.3)
        ax1.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='gray', alpha=0.1)
        ax1.set_title(f"[{ticker}] Bollinger Bands Analysis")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # [거래량]
        colors = ['red' if x > y else 'blue' for x, y in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
        ax2.set_title("Volume Flow")
        ax2.grid(True, alpha=0.3)

        # [RSI]
        ax3.plot(df.index, df['RSI'], color='purple', label='RSI(14)')
        ax3.axhline(70, color='red', linestyle='--')
        ax3.axhline(30, color='green', linestyle='--')
        ax3.set_title("RSI (Momentum)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Streamlit 출력
        st.pyplot(fig)
        st.success(f"✅ {ticker} 분석이 완료되었습니다.")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")

# 실행
if st.button("실시간 분석 시작"):
    analyze_ultimate_st(target_ticker)
