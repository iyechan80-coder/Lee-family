import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import asyncio
from telegram import Bot

# 1. 환경 설정
TELEGRAM_TOKEN = '7727210141:AAFWVsocDE_wm3zMYZKKJbP81d4XKydJZ0I' # BotFather에게 받은 토큰
CHAT_ID = '7555017085'         # 본인의 텔레그램 ID
MY_STOCKS = ['NVDA', '005930.KS', 'AAPL', 'MSFT', 'TSLA'] # 분석 대상 종목

# 2. 포트폴리오 최적화 함수
def optimize_portfolio(tickers):
    data = yf.download(tickers, period="2y")['Close']
    returns = data.pct_change().dropna()
    ann_returns = returns.mean() * 252
    ann_cov = returns.cov() * 252

    def objective(weights):
        p_ret = np.sum(ann_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov, weights)))
        return -p_ret / p_vol # Sharpe Ratio 최대화

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(len(tickers)))
    res = minimize(objective, [1/len(tickers)]*len(tickers), method='SLSQP', bounds=bnds, constraints=cons)
    return dict(zip(tickers, res.x.round(4)))

# 3. 텔레그램 보고서 전송
async def main():
    weights = optimize_portfolio(MY_STOCKS)
    
    report = "🚀 **오늘의 퀀트 전략 보고서**\n\n"
    report += "⚖️ **수학적 최적 비중 (Sharpe Ratio)**\n"
    for t, w in weights.items():
        if w > 0: report += f"- {t}: {w*100:.1f}%\n"
        
    report += "\n🌟 **AI Discovery 추천 후보**\n"
    report += "- 팔란티어(PLTR): AI 플랫폼 성장세 뚜렷\n" # AI 테마 반영
    report += "- 유나이티드헬스(UNH): 디지털 헬스케어 대장주\n" # 헬스케어 반영
    
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=report, parse_mode='Markdown')

if __name__ == "__main__":
    asyncio.run(main())