import sys, imp
imp.reload(sys.modules['KerasModelManager'])
imp.reload(sys.modules['models.models'])
imp.reload(sys.modules['PandasDataManager'])
from PandasDataManager import PandasDataManager
from KerasModelManager import KerasModelManager
# from models.models import l4_mp2_reg
import tensorflow as tf
from constants import target_name, x_path, y_path, x_test_path, y_test_path, preprocessed_data, x_reduced_path
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pandas as pd
import os

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
tf.random.set_seed(7)


class ProcessManager:
    # def __init__(self):

    def __init__(self):
        self.loss = None
        self.normalize = None
        self.mae = None
        self.mae_names = None
        self.reduce = None
        self.rescalex = None
        self.train_data_from_beginning = None
        self.test_perc = None
        self.train_perc = None
        self.models = None
        print("Process Manager Ready.")

    def run(self, models
            , train_perc=0.65
            , test_perc=0.10
            , train_data_from_beginning=True
            , rescalex=True
            , reduce=None
            , normalize=None
            , loss='mse'
            , metrics=[tf.keras.losses.MeanAbsoluteError()]
            , callbacks=None
            ):
        self.reduce = reduce
        self.normalize = normalize
        self.rescalex = rescalex
        self.train_data_from_beginning = train_data_from_beginning
        self.test_perc = test_perc
        self.train_perc = train_perc
        self.models = models
        self.loss = loss
        self.metrics = metrics
        self.mae_names = []
        self.mae = []

        pdm = PandasDataManager()
        X_train, X_valid, y_train, y_valid, Xt, yt = pdm.load_split_train_valid_test(path=preprocessed_data
                                                                                     , target_name=target_name
                                                                                     , train_perc=self.train_perc
                                                                                     , test_perc=self.test_perc
                                                                                     , train_data_from_beginning=self.train_data_from_beginning
                                                                                     , rescalex=self.rescalex
                                                                                     , reduce=self.reduce
                                                                                     , normalize=self.normalize
                                                                                     )
        print('X_train:', X_train.shape)
        print('X_valid:', X_valid.shape)
        print('Xt:', Xt.shape)

        for m in self.models:
            start_time = time.time()

            km = KerasModelManager(m['model'])
            km.compile_fit(X_train, y_train, X_valid, y_valid
                           , batch_size=m['batch']
                           , epochs=m['epochs']
                           , optimizer=tf.keras.optimizers.Adam()
                           , loss=self.loss
                           , metrics=self.metrics
                           , steps_per_epoch=m['steps_per_epoch']
                           , validation_steps=m['val_steps']
                           , validation_batch_size=m['val_batch']
                           , callbacks=callbacks
                           , verbose=1)

            km.plot_training_history()
            res = km.predict(Xt, yt)
            km.plot_predictions()
            # self.mae.append(res)
            # self.mae_names.append(m['name'])
            print('\n> Model:', m['name'])
            print("For Test Data: ", Xt.shape)
            print("- MAE: ", res['mae'])
            print("- MAE %: ", res['mape'])
            print("- MSE: ", res['mse'])
            print("- RMSE: ", res['rmse'])
            print(" [time secs: ", (time.time() - start_time), "]")

        # self.mae_names = mae_names
        # self.mae = mae
        # print("MAE: ", self.mae)
        print("DONE.")
        return self.mae, self.mae_names

    def plot_models_results(self, mae_names, mae):
        style = dict(size=15, color='black')
        mpl.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title("MAE for models", color='C0')
        for i in range(len(mae)):
            ax.text(i, mae[i], mae_names[i], **style)
        ax.plot(range(len(mae)), mae, color="#00f", label="mae")
        plt.legend()

    def ping(self):
        os.system("say -v Zosia 'Halo halo, Szefie. Zrobione.'")
