# config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# CoinMarketCap API
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")

# Reddit API
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")

# Any other project-specific constants can go here
BASE_COINMARKETCAP_URL = "https://pro-api.coinmarketcap.com/v1"
COINMARKETCAP_URL_LATEST = (
    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
)
COINMARKETCAP_URL_HISTORICAL = (
    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
)
