import json
import requests
import time
import logging
from LLM import extract_investor_info
from PortfolioBuilder import (
    getTickerGroup,
    choosingStocks,
    calculate_risk_tolerance_score,
    calc_amount_of_stock_to_buy,
)

# =============================
# Logging Configuration
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================
# API Configuration
# =============================
URL = "http://mts-prism.com"
PORT = 8082
TEAM_API_CODE = "Team_api_code" #Use team api here

# =============================
# API Helper Functions
# =============================
def send_get_request(path):
    headers = {"X-API-Code": TEAM_API_CODE}
    try:
        response = requests.get(f"{URL}:{PORT}/{path}", headers=headers)
        response.raise_for_status()
        return True, response.text
    except requests.RequestException as e:
        logger.error(f"GET request failed: {e}")
        return False, str(e)

def send_post_request(path, data=None):
    headers = {"X-API-Code": TEAM_API_CODE, "Content-Type": "application/json"}
    try:
        response = requests.post(f"{URL}:{PORT}{path}", data=json.dumps(data), headers=headers)
        response.raise_for_status()
        return True, response.text
    except requests.RequestException as e:
        logger.error(f"POST request failed: {e}")
        return False, str(e)

def get_context():
    return send_get_request("/request")

def get_my_current_information():
    return send_get_request("/info")

def send_portfolio(weighted_stocks):
    data = [
        {"ticker": ws[0], "quantity": float(ws[1])} if ws[0] == "CASH" else {"ticker": ws[0], "quantity": int(ws[1])}
        for ws in weighted_stocks
    ]
    logger.info(f"Submitting portfolio: {data}")
    return send_post_request("/submit", data=data)

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    success, info = get_my_current_information()
    if not success:
        logger.error(f"Failed to get team information: {info}")
        exit(1)

    logger.info(f"Team Info: {info}")

    while True:
        t0 = time.time()
        try:
            success, context = get_context()
            if not success:
                logger.warning(f"Error getting context: {context}")
                time.sleep(10)
                continue

            logger.info(f"Raw Context: {context}")
            t1 = time.time()

            investor_data = extract_investor_info(context)
            logger.info(f"Extracted Investor Data: {json.dumps(investor_data, indent=2, ensure_ascii=False)}")

            # Generate portfolio based on extracted investor data
            potential_list = getTickerGroup(investor_data["start_date"], investor_data["end_date"])
            selected_stocks = choosingStocks(potential_list, investor_data["start_date"], investor_data["end_date"], avoid_sectors=[investor_data.get("avoid", "")])
            risk_score = calculate_risk_tolerance_score(investor_data["age"], investor_data["salary"], investor_data["budget"])

            portfolio = calc_amount_of_stock_to_buy(
                selected_stocks,
                risk_score,
                investor_data["start_date"],
                investor_data["end_date"],
                investor_data["budget"]
            )

            if not portfolio:
                logger.warning("No valid stocks found that meet criteria.")
            else:
                success, response = send_portfolio(portfolio)
                if not success:
                    logger.error(f"Error submitting portfolio: {response}")
                logger.info(f"Evaluation Response: {response}")

            t2 = time.time()
            logger.info(f"Context processing time: {t1 - t0:.2f}s")
            logger.info(f"Total iteration time: {t2 - t0:.2f}s")

        except Exception as e:
            logger.critical(f"Critical error in iteration: {e}", exc_info=True)

        logger.info("Waiting for next iteration...")
        time.sleep(10)  # Configurable delay