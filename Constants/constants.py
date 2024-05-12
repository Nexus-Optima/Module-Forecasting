import os


class Commodities:
    COMMODITIES = "b3llcurve-commodities"
    FORECAST_STORAGE = "b3llcurve-forecast-storage"
    COTTON = "cotton"


class Credentials:
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")


class News:
    API_KEY = os.getenv("NEWS_API_KEY")
