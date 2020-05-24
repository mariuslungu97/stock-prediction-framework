"""
============IMPORTS============
"""
import numpy as np
from sklearn import preprocessing
import random
import pandas as pd

class DataProcess:
    """
    Class that provides abstractions for:
        1. Splits df into training, validation and testing chunks
        2. Splits training, validation and testing chunks into independent sequences of data
        3. Splits sequences of data into features and labels
        4. Processes each sequence of data using one of the following scalers: 'standard' | 'minmax' | 'maxabs'
        5. Balances the count of data points with different discrete labels to the one with the smallest size
    """
    def __init__(self, df, split_rate, is_clf, target_col=None, columns=None, validation_rate=None):

        df = df.dropna()
        df = df.sort_index(ascending=True)

        # if classification, regard label column as target
        if is_clf:
            df["target"] = df["label"]
            df = df.drop(columns=["label"])
            columns = ["target"] + [col for col in df.columns if col != "target"]
        else:
            # set target column (first column in columns) as the last column in the df
            if target_col:
                target_col = target_col.lower()
                df["target"] = df[target_col]
                df = df.drop(columns=[target_col])
                columns = ["target"] + [col for col in df.columns if col != "target"]
            elif columns:
                columns = [col.lower() for col in columns]
                df["target"] = df[columns[0]]
                df = df.drop(columns=[columns[0]])
                columns[0] = "target"

        # split the df into training and testing numpy representations, based on split_rate and selected columns
        split_index = int(len(df) * split_rate)
        self.data_train = df.get(columns).values[:split_index]

        if validation_rate:
            # split training data into training data and validation data
            # validation data will be the last %validation_rate of the training data
            validation_split = len(self.data_train) - int(len(self.data_train) * validation_rate)
            self.data_validation = self.data_train[validation_split:]
            self.data_train = self.data_train[:validation_split]

        self.data_test = df.get(columns).values[split_index:]

        self.train_len = len(self.data_train)
        self.test_len = len(self.data_test)
        self.validation_len = len(self.data_validation) if validation_rate else None

        self.is_clf = is_clf

    def get_train_data(self, seq_len=None, shuffle=None, processing_type=None):
        """
        :param seq_len: used to form limited windows of data for training
        :param shuffle: randomly shuffle the windows' order before returning the features and labels
        :param processing_type: used to process the windows
        :return: numpy arrays containing the processed features and labels of the training data
        """
        if shuffle is None:
            shuffle = False

        if processing_type is None:
            processing_type = "minmax"

        if seq_len is None:
            seq_len = 50

        data_shuffle = []
        data_x = []
        data_y = []

        # form windows (normalized)
        for i in range(self.train_len - seq_len):
            window = self._form_window(i, seq_len, processing_type)
            data_shuffle.append(window)

        # shuffle data
        if shuffle:
            random.shuffle(data_shuffle)

        # split windows into x and y
        for window in data_shuffle:

            x = window[:-1, 1:]
            y = window[-1, [0]]  # the value in the first col of the last window = the label

            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len=None, batch_size=None, shuffle=False, processing_type=None):
        """
        Can be used to avoid loading large datasets into memory at once.
        The method will load and process batches of sequences into memory, depending on batch_size
        """
        if seq_len is None:
            seq_len = 50

        if batch_size is None:
            batch_size = 64

        i = 0

        while i < (self.train_len - seq_len):
            batch = []
            x_batch = []
            y_batch = []

            for b in range(batch_size):
                if i >= self.train_len - seq_len:
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0

                window = self._form_window(i, seq_len, processing_type)
                batch.append(window)

                if shuffle:
                    random.shuffle(batch)

                x = batch[:, :-1, 1:]
                y = batch[:, -1, [0]]  # the value in the first col of the last window = the label
                x_batch.append(x)
                x_batch.append(y)
                i += 1

            yield np.array(x_batch), np.array(y_batch)

    def get_test_data(self, seq_len=None, processing_type=None):
        if processing_type is None:
            processing_type = "minmax"

        if seq_len is None:
            seq_len = 50

        data_windows = []

        for i in range(self.test_len - seq_len):
            window = self.data_test[i:i+seq_len]
            data_windows.append(window)

        data_windows = np.array(data_windows).astype(float)

        data_windows_processed = self._normalise_window(data_windows, single_window=False) \
            if processing_type else data_windows

        data_x = data_windows_processed[:, :-1, 1:]
        data_y = data_windows_processed[:, -1, [0]]

        return np.array(data_x), np.array(data_y)

    def get_validation_data(self, seq_len=None, shuffle=None, processing_type=None):

        if shuffle is None:
            shuffle = False

        if processing_type is None:
            processing_type = "minmax"

        if seq_len is None:
            seq_len = 50

        if self.data_validation is not None:
            data_windows = []

            for i in range(self.validation_len - seq_len):
                window = self.data_validation[i:i+seq_len]
                data_windows.append(window)

            if shuffle:
                random.shuffle(data_windows)

            data_windows = np.array(data_windows).astype(float)

            data_windows_processed = self._normalise_window(data_windows, single_window=False) \
                if processing_type else data_windows

            data_x = data_windows_processed[:, :-1, 1:]
            data_y = data_windows_processed[:, -1, [0]]

            return np.array(data_x), np.array(data_y)
        else:
            return None

    def balance_data(self, data):
        """
        Balances the count of data points with different discrete labels to the one with the smallest size
        WARNING: can considerably shrink the size of the data
        """
        buys, sells, holds = [], [], []

        for row in data:
            if row[-1] == 2:
                buys.append(row)
            elif row[-1] == 0:
                holds.append(row)
            elif row[-1] == 1:
                sells.append(row)

        lower = min(len(buys), len(sells), len(holds))

        buys = buys[:lower]
        holds = holds[:lower]
        sells = sells[:lower]

        data = buys + holds + sells

        return data

    def _form_window(self, i, seq_len, processing_type):

        window = self.data_train[i:i+seq_len]

        window = self._normalise_window(window, single_window=True)[0] if processing_type else window

        return window

    def _normalise_window(self, window_data, single_window=False):

        normalised_data = []
        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalised_window = []

            if self.is_clf:
                clf_col = window[:, 0]
                window = window[:, 1:]
            # normalize windows
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1)
                                  if float(window[0, col_i]) != 0 else float(p) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T

            if self.is_clf:
                normalised_window = np.insert(normalised_window, 0, values=clf_col, axis=1)

            normalised_data.append(normalised_window)

        return np.array(normalised_data)

    def _process_window(self, window_data, processing_type, single_window):

        data = []
        window_data = [window_data] if single_window else window_data

        if processing_type == "standard":
            scaler = preprocessing.StandardScaler()
        elif processing_type == "minmax":
            scaler = preprocessing.MinMaxScaler()
        elif processing_type == "maxabs":
            scaler = preprocessing.MaxAbsScaler()
        else:
            raise TypeError("Processing type is not supported! Please choose one of the following values: "
                            "'standard' | 'minmax' | 'maxabs'")

        for window in window_data:

            normalised_window = []
            clf_col = None

            if self.is_clf:
                clf_col = window[:, 0]
                window = window[:, 1:]

            # turn columns into ROC (rate of change) columns
            for col in range(window.shape[1]):
                normalised_col = \
                    [float(p) / float(window[row_idx - 1, col]) - 1 if window[row_idx - 1, col] != 0 else float(p)
                     for row_idx, p in enumerate(window[:, col])
                     ]
                normalised_window.append(normalised_col)

            # normalize/standardise columns, based on chosen scaler
            normalised_window = np.array(normalised_window).T
            # print(normalised_window.shape)
            normalised_window = scaler.fit_transform(normalised_window)

            if self.is_clf:
                normalised_window = np.insert(normalised_window, 0, values=clf_col, axis=1)

            data.append(normalised_window)

        return np.array(data)



