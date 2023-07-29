import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import metrics
import numpy as np


class KerasModelManager:
    def __init__(self, model):
        self.y_valid = None
        self.X_valid = None
        self.model = model
        self.X_train = None
        self.y_train = None
        self.history = None
        self.y_test = None
        self.y_pred = None

    def compile_fit(self, X_train, y_train
                    , X_valid, y_valid
                    , batch_size=64
                    , epochs=100
                    , verbose=0
                    , optimizer=tf.keras.optimizers.Adam()
                    , loss=tf.keras.losses.MeanAbsoluteError()
                    , metrics=[tf.keras.losses.MeanAbsoluteError()]
                    , steps_per_epoch=None
                    , validation_steps=None
                    , validation_batch_size=None
                    , callbacks=None
                    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.model.compile(
            optimizer=optimizer
            , loss=loss
            , metrics=metrics
        )

        # print("X_train: ", self.X_train.shape)
        # print("y_train: ", self.y_train.shape)
        # print("X_valid: ", self.X_valid.shape)
        # print("y_valid: ", self.y_valid.shape)
        #
        # steps per epoch
        self.history = self.model.fit(x=self.X_train, y=self.y_train
                                      , validation_data=(self.X_valid, self.y_valid)
                                      , batch_size=batch_size
                                      , epochs=epochs
                                      , verbose=verbose
                                      , steps_per_epoch=steps_per_epoch
                                      , validation_steps=validation_steps
                                      , validation_batch_size=validation_batch_size
                                      , callbacks=callbacks
                                      )
        # print("Model Summary: ", self.model.summary())

    def plot_training_history(self, x=None, y=None, labelx="Validation Loss", labely="Loss"):
        history_df = pd.DataFrame(self.history.history)

        print("Mean Validation Loss: {:0.5f}".format(history_df['val_loss'].mean()))

        X = history_df.val_loss if x is None else x
        Y = history_df.loss if y is None else y

        mpl.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title("Loss", color='C0')
        ax.plot(range(len(X)), X, color="#f00", label=labelx)
        ax.plot(range(len(Y)), Y, color="#00f", label=labely)
        plt.legend()

    def plot_predictions(self):
        mpl.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title("Baseline vs Predictions", color='C0')
        ax.plot(range(len(self.y_test)), self.y_test, color="#f00", label="Baseline")
        ax.plot(range(len(self.y_pred)), self.y_pred, color="#00f", label="Predictions")
        plt.legend()

    def predict(self, X, y):
        self.y_test = y
        self.y_pred = self.model.predict(X)
        res = {}
        # mae = metrics.mean_absolute_error(true, predicted)
        res['mse'] = metrics.mean_squared_error(y, self.y_pred)
        res['rmse'] = np.sqrt(metrics.mean_squared_error(y, self.y_pred))
        # res['r2_square'] = metrics.r2_score(y, self.y_pred)
        res['mae'] = metrics.mean_absolute_error(y, self.y_pred)
        res['mape'] = metrics.mean_absolute_percentage_error(y, self.y_pred)
        return res

    def predict_report(self, X, y, plot_predictions=False):
        self.y_test = y
        self.y_pred = self.model.predict(X)

        print('MAE :', metrics.mean_absolute_error(y, self.y_pred))
        print('MAE % :', metrics.mean_absolute_percentage_error(y, self.y_pred))
        if plot_predictions:
            self.plot_predictions()
