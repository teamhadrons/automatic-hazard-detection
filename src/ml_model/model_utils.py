from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import datetime
from pandas import read_csv
# prepare data for lstm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from numpy import array
import os


class ModelUtils():
    def parse_date(x):
        return datetime.strptime(x, '%Y %m %d %H')

    @staticmethod
    def save_clean_csv_dataset(raw_csv_dataset_path: str = None):
        dataset = read_csv(raw_csv_dataset_path,
                           parse_dates=[['year', 'month', 'day', 'hour']],
                           index_col=0, date_parser=ModelUtils.parse_date)

        dataset.drop('No', axis=1, inplace=True)
        # manually specify column names
        dataset.columns = ['pollution', 'dew', 'temp',
                           'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'
        # mark all NA values with 0
        dataset['pollution'].fillna(0, inplace=True)
        now = datetime.now()  # current date and time
        timestamp_str = now.strftime("%m-%d-%Y-%H-%M-%S-%f")
        dataset_name = f"dataset_{timestamp_str}.csv"
        dataset_path = os.path.join('assets', 'datasets', dataset_name)
        # save to file
        dataset.to_csv(dataset_path)
        return dataset_path

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """convert series to supervised learning"""
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    def prepare_data(series, n_test, n_lag, n_seq):
        # extract raw values
        values = series.values
        # integer encode direction
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # transform into supervised learning problem X, y
        supervised = ModelUtils.series_to_supervised(
            scaled, n_lag, n_seq)
        supervised_values = supervised.values
        # split into train and test sets
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return scaler, train, test

    def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:n_lag], train[:, n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # design network
        model = Sequential()
        model.add(LSTM(n_neurons, batch_input_shape=(
            n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X, y, epochs=nb_epoch, batch_size=n_batch,
                  verbose=1, shuffle=False)
        return model

    def forecast_lstm(model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            print(y)
            # make forecast
            forecast = ModelUtils.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    def inverse_difference(last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted

    def inverse_transform(series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = ModelUtils.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted
