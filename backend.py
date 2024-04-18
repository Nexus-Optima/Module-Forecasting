from flask import Flask, request, jsonify
import matplotlib

from Execute.execute import forecast_pipeline
from Database.s3_operations import read_forecast
from News_Insights.news import fetch_news

import threading

app = Flask(__name__)

matplotlib.use("Agg")


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        commodity_name = data.get('commodity_name')
        if not commodity_name:
            return jsonify({"error": "commodity_name is required in the request body"}), 400
        thread = threading.Thread(target=forecast_pipeline, args=commodity_name)
        thread.start()
        return jsonify({"message": "Forecasting started for " + commodity_name}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-forecast/<commodity_name>', methods=['GET'])
def get_forecast(commodity_name):
    try:
        if not commodity_name:
            return jsonify({"error": "commodity_name is required as a URL parameter"}), 400

        forecast_data = read_forecast(commodity_name)
        return jsonify({
            "actual": forecast_data['actual'].to_dict(orient='records'),
            "forecast": forecast_data['forecast'].to_dict(orient='records')
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_news_by_commodity/<commodity_name>', methods=['GET'])
def get_news_by_commodity(commodity_name):
    if not commodity_name:
        return jsonify({'error': 'Commodity name is required'}), 400

    try:
        news_data = fetch_news(commodity_name)
        return jsonify(news_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
