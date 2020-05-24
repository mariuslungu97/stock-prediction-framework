"""
============IMPORTS============
"""
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import os
import numpy as np
from numpy import newaxis
import datetime as dt


class NNModel:
    """
    Provides abstractions for:
        1. builds lstm nn model, based on a config file
        2. trains nn model using X and y data - either by using a generator or by loading the entire data into memory
        3. predicts y point-by-point, using real past data
        4. predicts y by sequence, using previously predicted data; restarts with real data once sequence finished
        5. predicts y by only using previously predicted data

    The class momentarily only supports LSTM based neural networks.
    """
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print("[NNModel] Loading model from file %s" % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, input_seq, input_features):
        layer_count = 0
        for layer in configs["nn_model"]["layers"]:
            neurons = layer["neurons"] if "neurons" in layer else None
            dropout_rate = layer["rate"] if "rate" in layer else None
            activation = layer["activation"] if "activation" in layer else None
            return_seq = layer["return_seq"] if "return_seq" in layer else None

            if layer["type"] == "dense":
                if layer_count == 0:
                    self.model.add(Dense(neurons, input_shape=(input_seq, input_features), activation=activation))
                else:
                    self.model.add(Dense(neurons, activation=activation))
            if layer["type"] == "lstm":
                if layer_count == 0:
                    self.model.add(LSTM(neurons, input_shape=(input_seq, input_features), return_sequences=return_seq))
                else:
                    self.model.add(LSTM(neurons, return_sequences=return_seq))
            if layer["type"] == "dropout":
                self.model.add(Dropout(dropout_rate))
            if layer["type"] == "batch_normalization":
                self.model.add(BatchNormalization())

            layer_count += 1

        self.model.compile(
            loss=configs["nn_model"]["loss"] if "loss" in configs["nn_model"] else "mse",
            optimizer=configs["nn_model"]["optimizer"] if "optimizer" in configs["nn_model"] else "adam",
            metrics=configs["nn_model"]["metrics"] if "metrics" in configs["nn_model"] else None
        )

        print("[NNModel] Model Compiled")

    def train(self, x, y, epochs, batch_size, validation_x=None, validation_y=None):
        print("[NNModel] Training Started")
        print("[NNModel] %s epochs, %s batch size" % (epochs, batch_size))

        save_fname = os.path.join("saved_models", '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        logs_fname = os.path.join("logs", "fit", '%s-e%s' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2, verbose=1),
            ModelCheckpoint(filepath=save_fname, monitor="val_loss", save_best_only=True),
            TensorBoard(log_dir=logs_fname)
        ]

        self.model.summary()
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(validation_x, validation_y)
        )
        self.model.save(save_fname)

        print('[NNModel] Training Completed. Model saved as %s' % save_fname)

    def train_generator(self, data_gen, epochs, batch_size, validation_x=None, validation_y=None):

        print('[NNModel] Training Started')
        print('[NNModel] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join("saved_models",
                                  '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        logs_fname = os.path.join("logs", "fit", '%s-e%s' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=logs_fname)
        ]

        self.model.summary()
        self.model.fit_generator(
            data_gen,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
            validation_data=(validation_x, validation_y)
        )
        self.model.save(save_fname)

        print('[NNModel] Training Completed. Model saved as %s' % save_fname)

    def predict_point_by_point(self, data, classify=False):
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size, ))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        print('[NNModel] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        print('[NNModel] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
