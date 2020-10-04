from flask import Flask
from src.ml_model.model_utils import ModelUtils
from flask import jsonify

app = Flask(__name__)


@app.route('/forecast/<steps>')
def predict_weather(steps: int):
    # lloading this data from a server :)
    train = []
    test = []
    forecasts = ModelUtils.make_forecasts(train, test, 1, steps)
    return jsonify(forecasts)


if __name__ == '__main__':
    app.run()
