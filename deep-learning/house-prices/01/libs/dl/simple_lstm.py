from libs.dl.simple_regressor import SimpleRegressor
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


class SimpleLSTM(SimpleRegressor):
    def __init__(self, X_train, y_train, X_valid, y_valid):
        # self.model = None
        SimpleRegressor.__init__(self, X_train, y_train, X_valid, y_valid)

    def build_model(self, units, activation, layers_count):
        print('Building LSTM model...')
        m = Sequential()
        m.add(LSTM(units=units, activation=activation, input_shape=[self.features_count, 1]))
        for i in range(layers_count):
            m.add(Dense(units=units, activation=activation))

        m.add(Dense(1))
        self.model = m
