from flask import Flask
from src.ml_model.model_utils import ModelUtils
from flask import jsonify
from pandas import read_csv

app = Flask(__name__)


@app.route('/forecast/<steps>')
def predict_weather(steps: int):
    # load dataset
    dataset_path = ModelUtils.save_clean_csv_dataset(
        'assets/raw.csv')
    series = read_csv(dataset_path, header=0, index_col=0)
    # configure
    n_lag = 1
    n_seq = int(steps)
    n_test = 10
    n_epochs = 1
    n_batch = 1
    n_neurons = 1
    # prepare data
    scaler, train, test = ModelUtils.prepare_data(series, n_test, n_lag, n_seq)
    # fit model
    model = ModelUtils.fit_lstm(
        train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    # make forecasts
    forecasts = ModelUtils.make_forecasts(
        model, n_batch, train, test, n_lag, n_seq)
    # inverse transform forecasts and test
    forecasts = [row[n_lag:] for row in test]
    return jsonify([forecast.tolist() for forecast in forecasts])


if __name__ == '__main__':
    app.run()
