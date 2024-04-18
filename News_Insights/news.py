import requests
from Constants.constants import News


def fetch_news(commodity):
    """
    Fetches news from the Newsdata.io API related to a specified commodity.

    Parameters:
    - api_key: The API key to authenticate the request.
    - commodity: The commodity name to filter news by.

    Returns:
    - A JSON object containing news articles related to the specified commodity.
    """
    url = "https://newsdata.io/api/1/news"
    params = {
        'apikey': News.API_KEY,
        'q': commodity  # Using 'q' parameter to filter news by the commodity name
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to fetch news', 'status_code': response.status_code}
