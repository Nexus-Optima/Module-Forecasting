from flask import Flask, jsonify
import matplotlib
from Execute.execute import forecast_pipeline
import threading

app = Flask(__name__)

matplotlib.use("Agg")


@app.route('/forecast', methods=['GET'])
def forecast():
    try:
        thread = threading.Thread(target=forecast_pipeline)
        thread.start()
        return jsonify({"message": "Forecasting started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
