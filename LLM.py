import requests
import json
import re
import logging
from typing import Dict, Optional

# =============================
# Logging Configuration
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================
# Investor Info Extraction Function
# =============================
def extract_investor_info(text: str) -> Dict[str, Optional[float]]:
    """
    Extracts investor details from input text using a local LLM API.
    
    Args:
        text (str): Raw context text containing investor information.

    Returns:
        dict: Extracted investor details with defaults where necessary.
    """
    required_fields = ["age", "budget", "start_date", "end_date", "avoid", "salary"]
    default_values = {
        "age": 30,
        "budget": 100000.00,
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
        "avoid": "",
        "salary": 50000.00
    }

    prompt = (
        "Analyze this text and extract the following details as a valid JSON object:\n"
        "{\n"
        "    \"age\": <integer>,\n"
        "    \"budget\": <decimal number with optional cents>,\n"
        "    \"start_date\": <YYYY-MM-DD>,\n"
        "    \"end_date\": <YYYY-MM-DD>,\n"
        "    \"avoid\": <text description>,\n"
        "    \"salary\": <decimal number if present>\n"
        "}\n\n"
        f"Text to analyze: \"{text}\"\n\n"
        "Return ONLY the JSON object with double quotes, nothing else."
    )

    try:
        # Check if LLM server is running
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        logger.info("LLM API is reachable.")

        # Send prompt to local LLM
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3}
            },
            timeout=30
        ).json()

        raw_response = response.get("response", "").strip()

        # Extract JSON block safely using regex
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            logger.warning("No valid JSON found in LLM response. Using default values.")
            return default_values

        data = json.loads(json_match.group())

        # Convert numeric fields safely with fallbacks
        conversions = {
            'age': lambda x: int(x) if x else default_values['age'],
            'budget': lambda x: round(float(str(x).replace(',', '')), 2) if x else default_values['budget'],
            'salary': lambda x: round(float(str(x).replace(',', '')), 2) if x else default_values['salary']
        }

        investor_info = {
            field: conversions[field](data.get(field, default_values[field]))
            if field in conversions else data.get(field, default_values[field])
            for field in required_fields
        }

        logger.info(f"Extracted investor info: {investor_info}")
        return investor_info

    except requests.RequestException as api_error:
        logger.error(f"LLM API connection error: {api_error}")
    except (json.JSONDecodeError, ValueError) as parsing_error:
        logger.error(f"Error parsing LLM response: {parsing_error}")
    except Exception as e:
        logger.exception(f"Unexpected error during extraction: {e}")

    return default_values
