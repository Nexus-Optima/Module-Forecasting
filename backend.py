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


if __name__ == '__main__':
    app.run(debug=True)
