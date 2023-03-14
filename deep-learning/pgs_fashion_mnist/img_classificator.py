import warnings
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

warnings.filterwarnings('ignore')


class SimpleImgClassificatorManager:
    def __init__(self, train_df, test_df, class_names, tf_loglevel='3'):
        self.predicted_classes = None
        self.model = None
        self.train_df = train_df
        self.test_df = test_df
        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None
        self.x_test = None
        self.y_test = None
        self.class_names = class_names
        self.image_shape = None
        self.batch_size = None
        self.fit_history = None
        self.score = None

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_loglevel
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = tf_loglevel

    def preprocess_image_data(self, test_size=0.2):
        start = time.time()
        num_classes = len(self.class_names)
        train_np = np.array(self.train_df, dtype='float32')
        test_np = np.array(self.test_df, dtype='float32')
        # Since the image data in x_train and x_test is from 0 to 255 ,  we need to rescale this from 0 to 1.
        # To do this we need to divide the x_train and x_test by 255 . It's important that the training set and
        # the testing set be preprocessed in the same way:
        x_train = train_np[:, 1:] / 255
        y_train = train_np[:, 0]
        x_test = test_np[:, 1:] / 255
        y_test = test_np[:, 0]
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=test_size,
                                                                    random_state=12345)
        image_rows = 28
        image_cols = 28
        # batch_size = 4096
        # the shape of the image as 3d with rows and columns grey scale
        self.image_shape = (image_rows, image_cols, 1)

        self.x_train = x_train.reshape(-1, *self.image_shape)
        self.x_test = x_test.reshape(-1, *self.image_shape)
        self.x_validate = x_validate.reshape(-1, *self.image_shape)
        self.y_train = to_categorical(y_train, num_classes)
        self.y_validate = to_categorical(y_validate, num_classes)
        self.y_test = to_categorical(y_test, num_classes)
        print('(Preprocessing: Done in {:.2f} secs)'.format(time.time() - start))

    def compile_fit_report(self, model, batch_size, epochs
                           , compile_optimizer=None
                           , fit_callbacks=None, img_data_generator=None):
        start = time.time()
        self.batch_size = batch_size

        model.compile(optimizer=compile_optimizer
                      , loss="categorical_crossentropy"
                      , metrics=["accuracy"])

        if img_data_generator is None:
            self.fit_history = model.fit(self.x_train, self.y_train
                                         , epochs=epochs
                                         , validation_data=(self.x_validate, self.y_validate)
                                         , validation_steps=len(self.x_validate) // batch_size
                                         , verbose=1
                                         , steps_per_epoch=len(self.x_train) // batch_size
                                         , callbacks=fit_callbacks)
        else:
            img_data_generator.fit(self.x_train)
            self.fit_history = model.fit(img_data_generator.flow(self.x_train, self.y_train, batch_size=batch_size)
                                         , epochs=epochs
                                         , validation_data=(self.x_validate, self.y_validate)
                                         , validation_steps=len(self.x_validate) // batch_size
                                         , verbose=1
                                         , steps_per_epoch=len(self.x_train) // batch_size
                                         , callbacks=fit_callbacks)
        self.model = model
        self.plot_fit_history()
        print('(Compile & Fit: Done in {:.2f} mins)'.format((time.time() - start) / 60))

    def evaluate_report(self):
        start = time.time()
        self.score = self.model.evaluate(self.x_test, self.y_test, verbose=1)

        print('===========================')
        print('>>>> Test Loss : {:.4f}'.format(self.score[0]))
        print('>>>> Test Accuracy : {:.4f}'.format(self.score[1]))
        print('===========================\n\n')
        print('(Evaluate & Report: Done in {:.2f} mins)'.format((time.time() - start) / 60))

    def predict_report(self):
        start = time.time()
        self.predicted_classes = np.argmax(self.model.predict(self.x_test), axis=-1)
        y_test = self.test_df.iloc[:, 0]
        print(classification_report(y_test, self.predicted_classes, target_names=self.class_names))
        print('(Predict & Report: Done in {:.2f} mins)'.format((time.time() - start) / 60))
        return y_test, self.predicted_classes

    def plot_fit_history(self):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.fit_history.history['loss'], color='b', label="Training loss")
        ax[0].plot(self.fit_history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(self.fit_history.history['accuracy'], color='b', label="Training accuracy")
        ax[1].plot(self.fit_history.history['val_accuracy'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)

    def plot_images(self):
        plt.figure(figsize=(10, 10))
        for i in range(36):
            plt.subplot(6, 6, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i].reshape((28, 28)))
            label_index = int(self.y_train[i])
            plt.title(self.class_names[label_index])
        plt.show()

    def plot_predictions(self):
        L = 5
        W = 5
        fig, axes = plt.subplots(L, W, figsize=(12, 12))
        axes = axes.ravel()

        for i in np.arange(0, L * W):
            axes[i].imshow(self.x_test[i].reshape(28, 28))
            # axes[i].set_title(f"Prediction Class = {predicted_classes[i]:0.1f}\n Original Class = {y_test[i]:0.1f}")
            axes[i].set_title(
                f"Predicted: {self.class_names[self.predicted_classes[i]]}\n Original: {self.class_names[self.y_test[i]]}")
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0.5)

    def image_data_generator(self):
        return ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    def learning_rate_reductor(self, patience=3, factor=0.5, min_lr=0.00001):
        return ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=patience,
                                 verbose=1,
                                 factor=factor,
                                 min_lr=min_lr)

    def create_model(self, filters=64, dropout=0.25, first_kernel=(5, 5), second_kernel=(3, 3), pool_size=(2, 2),
                     additionl_block=True):
        model = Sequential()
        model.add(Conv2D(filters=filters, kernel_size=first_kernel, padding='Same', activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=filters, kernel_size=first_kernel, padding='Same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=pool_size))
        model.add(Dropout(dropout))

        if additionl_block:
            model.add(Conv2D(filters=filters, kernel_size=second_kernel, padding='Same', activation='relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=filters, kernel_size=second_kernel, padding='Same', activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
            model.add(Dropout(dropout))

        model.add(Conv2D(filters=filters, kernel_size=second_kernel, padding='Same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(len(self.class_names), activation="softmax"))

        return model
