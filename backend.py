from flask import Flask, request, jsonify
import matplotlib
from Execute.execute import forecast_pipeline
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
