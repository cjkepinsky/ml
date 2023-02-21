from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner
import pandas as pd
import datetime


# early_stopping = EarlyStopping(
#     min_delta=0.001,  # minimium amount of change to count as an improvement
#     patience=7,  # how many epochs to wait before stopping
#     restore_best_weights=True
# )


class SimpleRegressor:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.model = None
        self.winner = None
        self.winner_params = None
        self.units_min = None
        self.units_max = None
        self.units_step = None
        self.features_count = self.X_train.shape[1]
        self.layers_count = 3

    def hyperband_search(self, max_epochs=5, batch_size=128, units_min=130, units_max=160,
                         units_step=4, layers_count=3):
        self.units_min = units_min
        self.units_max = units_max
        self.units_step = units_step
        self.layers_count = layers_count

        # tuner = keras_tuner.RandomSearch(
        tuner = keras_tuner.Hyperband(
            hypermodel=self.build_fit_callback,
            objective=keras_tuner.Objective("val_accuracy", direction="max"),
            max_epochs=50,
            factor=3,
            seed=42,
            overwrite=True,
            directory="logs/keras_tuner"
        )
        tuner.search(
            x=self.X_train,
            y=self.y_train,
            epochs=max_epochs,
            batch_size=batch_size,
            validation_data=(self.X_valid, self.y_valid)
        )
        tuner.results_summary()
        winner = tuner.get_best_models(num_models=1)[0]
        winner.build(input_shape=[self.features_count])
        self.winner_params = tuner.get_best_hyperparameters(1)
        self.winner = winner
        return winner

    def build_fit_callback(self, hp):
        activation = 'relu'
        units_tuner = hp.Int("units", min_value=self.units_min, max_value=self.units_max, step=self.units_step)
        # layers_count_tuner = hp.Int("units", min_value=min, max_value=max, step=self.units_step)

        self.build_model(units=units_tuner, activation=activation, layers_count=self.layers_count)
        self.compile_fit(epochs=10, batch_size=128)
        return self.model

    def build_model(self, units, activation, layers_count):
        m = Sequential()
        m.add(Dense(units=units, activation=activation, input_shape=[self.features_count]))
        for i in range(layers_count):
            m.add(Dense(units=units, activation=activation))

        m.add(Dense(1))
        self.model = m

    def compile_fit(self, epochs=400, batch_size=128):
        log_dir = "logs/tensorboard/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # metrics=[MeanAbsoluteError()],

        self.model.compile(
            optimizer="adam",
            loss="mean_absolute_error",
            metrics=['accuracy']
        )

        history = self.model.fit(x=self.X_train, y=self.y_train,
                                 validation_data=(self.X_valid, self.y_valid)
                                 , batch_size=batch_size
                                 , epochs=epochs
                                 # , callbacks=[early_stopping]
                                 , use_multiprocessing=True
                                 # , verbose='2'
                                 , callbacks=[tensorboard_callback]
                                 )
        return pd.DataFrame(history.history)

    def get_model(self):
        return self.model

    # m.compile(
    #     optimizer="adam",
    #     loss="mean_absolute_error",
    #     metrics=[keras.metrics.MeanAbsoluteError()],
    # )
    # # return m
    #
    # history = model.fit(x=self.X_train, y=self.y_train,
    #                     validation_data=(self.X_valid, self.y_valid)
    #                     , batch_size=128
    #                     , epochs=400
    #                     # , callbacks=[early_stopping]
    #                     , use_multiprocessing=True
    #                     # , verbose='2'
    #                     )
    # return pd.DataFrame(history.history)
