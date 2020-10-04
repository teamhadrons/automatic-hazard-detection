from pandas import read_csv
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import os


class ModelClient():
    def __init__():
        pass
    # load data

    def parse_date(x):
        return datetime.strptime(x, '%Y %m %d %H')

    @staticmethod
    def save_clean_csv_dataset(raw_csv_dataset_path: str = None):
        dataset = read_csv(raw_csv_dataset_path,
                           parse_dates=[['year', 'month', 'day', 'hour']],
                           index_col=0, date_parser=ModelClient.parse_date)

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

    @staticmethod
    def reframe_dataset(clean_dataset_path):
        # load dataset
        dataset = read_csv(clean_dataset_path, header=0, index_col=0)
        values = dataset.values
        # integer encode direction
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = ModelClient.series_to_supervised(scaled, 1, 1)
        # drop columns we don't want to predict
        reframed.drop(
            reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
        return reframed

    @staticmethod
    def train_model(reframe_dataset):
        # split into train and test sets
        values = reframe_dataset.values
        n_train_hours = 365 * 24
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        model.fit(train_X, train_y, epochs=50, batch_size=72,
                  validation_data=(test_X, test_y), verbose=2, shuffle=False)
        # plot history
        return model

    @staticmethod
    def predict_on_model(model, input):
        return model.predict(input)
