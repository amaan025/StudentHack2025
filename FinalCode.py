import pandas as pd
import numpy as np
import logging
import yfinance as yf
import requests
import json
import time
from io import StringIO
from typing import List, Tuple, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential

# =============================
# Logging Configuration
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================
# Constants & Configuration
# =============================
URL = "http://mts-prism.com"
PORT = 8082
TEAM_API_CODE = "Team_api_code" #Use team api here

CPI_DATA_FILE = 'cpi_index_all_00-25.csv'
TICKERS_INDEX = ["GSPC", "MSCI", "AAPL", "DJI"]

# =============================
# Load CPI Data
# =============================
def load_cpi_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=12, names=['Year'] + list(range(1, 13)) + ['Half1', 'Half2'])
    df.set_index('Year', inplace=True)
    logger.info("CPI data loaded successfully.")
    return df

cpi_data = load_cpi_data(CPI_DATA_FILE)

# =============================
# API Helpers
# =============================
def send_get_request(path: str) -> Tuple[bool, str]:
    headers = {"X-API-Code": TEAM_API_CODE}
    try:
        response = requests.get(f"{URL}:{PORT}/{path}", headers=headers)
        response.raise_for_status()
        return True, response.text
    except requests.RequestException as e:
        logger.error(f"GET request failed: {e}")
        return False, str(e)

def send_post_request(path: str, data: Optional[dict] = None) -> Tuple[bool, str]:
    headers = {"X-API-Code": TEAM_API_CODE, "Content-Type": "application/json"}
    try:
        response = requests.post(f"{URL}:{PORT}{path}", data=json.dumps(data), headers=headers)
        response.raise_for_status()
        return True, response.text
    except requests.RequestException as e:
        logger.error(f"POST request failed: {e}")
        return False, str(e)

# =============================
# Financial Data Helpers
# =============================
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def safe_yfinance_request(ticker: str) -> Optional[dict]:
    try:
        info = yf.Ticker(ticker).info
        time.sleep(0.2)  # Basic rate limiting
        return info
    except Exception as e:
        logger.warning(f"Failed to fetch yFinance data for {ticker}: {e}")
        return None

def stock_price(ticker: str, start: str, end: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            logger.warning(f"No price data found for {ticker}")
            return None, None
        return round(data.iloc[0]['Open'], 4), round(data.iloc[-1]['Open'], 4)
    except Exception as e:
        logger.error(f"Error fetching stock price for {ticker}: {e}")
        return None, None

# =============================
# Risk & Evaluation Functions
# =============================
def systematic_risk(ticker: str) -> float:
    info = safe_yfinance_request(ticker)
    return info.get("beta", 1.0) if info else 1.0

def liquidity_risk(ticker: str) -> bool:
    info = safe_yfinance_request(ticker)
    if info:
        return info.get("averageVolume", 0) > 100_000 or info.get("marketCap", 0) > 500_000
    return False

# =============================
# Portfolio Construction
# =============================
def calculate_risk_tolerance(age: int, salary: float, budget: float) -> float:
    age_factor = 1 - (age / 100)
    salary_factor = min(salary / 200_000, 1)
    budget_factor = min(budget / 50_000, 1)
    score = (0.4 * age_factor) + (0.3 * salary_factor) + (0.3 * budget_factor)
    logger.info(f"Calculated risk tolerance: {score:.2f}")
    return score

def build_portfolio(tickers: List[str], risk_tolerance: float, start: str, end: str, budget: float) -> List[Tuple[str, int]]:
    portfolio = []
    portfolio_data = []

    for ticker in tickers:
        start_price, end_price = stock_price(ticker, start, end)
        if not all([start_price, end_price]):
            continue

        investment_return = (end_price - start_price) / start_price
        beta = systematic_risk(ticker)

        if beta <= 0 or risk_tolerance <= 0:
            continue

        weight = abs(investment_return) / (risk_tolerance * beta)
        portfolio_data.append({"ticker": ticker, "weight": weight, "price": end_price})

    if not portfolio_data:
        logger.warning("No valid stocks for portfolio.")
        return portfolio

    total_weight = sum(item['weight'] for item in portfolio_data)

    for item in portfolio_data:
        shares = int((item['weight'] / total_weight * budget) // item['price'])
        if shares > 0:
            portfolio.append((item['ticker'], shares))

    logger.info(f"Constructed portfolio: {portfolio}")
    return portfolio

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    success, context = send_get_request("/request")
    if not success:
        logger.error(f"Failed to fetch client context: {context}")
        exit(1)

    client_data = json.loads(json.loads(context).get('message', '{}'))

    logger.info(f"Client data: {client_data}")

    tickers = TICKERS_INDEX

    risk_score = calculate_risk_tolerance(
        age=client_data.get("age", 30),
        salary=client_data.get("salary", 50_000),
        budget=client_data.get("budget", 10_000)
    )

    portfolio = build_portfolio(
        tickers,
        risk_score,
        client_data.get("start"),
        client_data.get("end"),
        client_data.get("budget", 10_000)
    )

    if not portfolio:
        logger.error("No portfolio constructed.")
        exit(1)

    success, response = send_post_request("/submit", data=[{"ticker": t, "quantity": q} for t, q in portfolio])
    if success:
        logger.info("Portfolio submitted successfully.")
    else:
        logger.error(f"Portfolio submission failed: {response}")
