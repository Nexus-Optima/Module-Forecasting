from flask import Flask, request, jsonify
import matplotlib
from flask_cors import CORS
from Utils.model_details import fetch_all_model_details
from Database.s3_operations import read_forecast
from News_Insights.news import fetch_news
import threading

application = Flask(__name__)
cors = CORS(application)
matplotlib.use("Agg")


def format_date(date):
    return date.strftime('%Y-%m-%d')


@application.route('/get-forecast/<commodity_name>/micro', methods=['GET'])
def get_forecast_micro(commodity_name):
    try:
        if not commodity_name:
            return jsonify({"error": "commodity_name is required as a URL parameter"}), 400

        forecast_data = read_forecast(commodity_name, 'micro')
        forecast_data['actual']['Date'] = forecast_data['actual']['Date'].apply(format_date)
        forecast_data['predictions']['Date'] = forecast_data['predictions']['Date'].apply(format_date)
        forecast_data['forecast']['Date'] = forecast_data['forecast']['Date'].apply(format_date)

        return jsonify({
            "actual": forecast_data['actual'].to_dict(orient='records'),
            "predictions": forecast_data['predictions'].to_dict(orient='records'),
            "forecast": forecast_data['forecast'].to_dict(orient='records'),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@application.route('/get-forecast/<commodity_name>/macro', methods=['GET'])
def get_forecast_macro(commodity_name):
    try:
        if not commodity_name:
            return jsonify({"error": "commodity_name is required as a URL parameter"}), 400

        forecast_data = read_forecast(commodity_name, 'macro')
        forecast_data['actual']['Date'] = forecast_data['actual']['Date'].apply(format_date)
        forecast_data['predictions']['Date'] = forecast_data['predictions']['Date'].apply(format_date)
        forecast_data['forecast']['Date'] = forecast_data['forecast']['Date'].apply(format_date)

        return jsonify({
            "actual": forecast_data['actual'].to_dict(orient='records'),
            "predictions": forecast_data['predictions'].to_dict(orient='records'),
            "forecast": forecast_data['forecast'].to_dict(orient='records'),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@application.route('/get_news_by_commodity/<commodity_name>', methods=['GET'])
def get_news_by_commodity(commodity_name):
    if not commodity_name:
        return jsonify({'error': 'Commodity name is required'}), 400

    try:
        news_data = fetch_news(commodity_name)
        formatted_data = []
        for article in news_data['results']:
            if 'title' in article:
                description = article.get('description', 'No description available')
                formatted_data.append({'title': article['title'], 'description': description})
        return jsonify(formatted_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@application.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


@application.route('/get_model_details', methods=['GET'])
def get_model_details():
    """Flask route to get model details from DynamoDB."""
    items = fetch_all_model_details()
    if items is not None:
        return jsonify(items)
    else:
        return jsonify({"error": "Failed to retrieve data"}), 500


if __name__ == '__main__':
    application.run(host="0.0.0.0", debug=True)
