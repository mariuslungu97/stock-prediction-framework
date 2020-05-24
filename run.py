"""
============IMPORTS============
"""
from src.fetch_data import DataFetcher
from src.load_data import DataLoader
from src.classify_data import DataClassifier
from src.visualise_data import DataVisualiser
from src.process_data import DataProcess
from src.nn_model import NNModel
from src.clf_model import CLFModel

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from os import path
import json
import pandas as pd
from collections import Counter
import datetime as dt
import sys

"""
============GLOBALS============
"""
pd.options.display.width = 0

credentials_path = path.join(path.dirname(path.abspath(__file__)), "credentials.json")
credentials_json = json.load(open(credentials_path, "r"))

configs_path = path.join(path.dirname(path.abspath(__file__)), "config.json")
configs_json = json.load(open(configs_path, "r"))


"""
============UTILS METHODS============
"""


def flatten_3d_data(data):
    # the classification algorithms expect data with shape = 2
    n_samples, windows, features = data.shape
    data = data.reshape((n_samples, windows * features))
    return data


"""
============MAIN METHODS============
"""


def fetch_data():

    # mandatory params
    av_api_key = credentials_json["av_api_key"]
    tickers = configs_json["fetch_data"]["tickers"]
    time_window = configs_json["fetch_data"]["time_window"]

    data_fetcher = DataFetcher(av_api_key)
    has_data_fetch = data_fetcher.fetch_data(tickers, time_window)
    return has_data_fetch


def load_data():

    # mandatory params
    data_type = configs_json["load_data"]["data_type"]
    tickers = configs_json["load_data"]["tickers"]

    # optional params
    tech_ind = configs_json["load_data"]["tech_indicators"] if "tech_indicators" in configs_json["load_data"] else None
    use_adjusted = bool(configs_json["load_data"]["use_adj"]) if "use_adj" in configs_json["load_data"] else None
    visualise_corr = bool(configs_json["load_data"]["visualise_corr"]) \
        if "visualise_corr" in configs_json["load_data"] else None
    visualise_tickers = configs_json["load_data"]["visualise_tickers"] \
        if "visualise_tickers" in configs_json["load_data"] else []

    data_loader = DataLoader(tickers, data_type, tech_ind)
    data = data_loader.load_data(use_adj=use_adjusted)

    if len(data) <= 0:
        print(f"No data using the following tickers: {tickers} has been found. Please try again using available tickers!")
        sys.exit(0)

    print("LOADED DATA HEAD\n", data.head(5))
    print("LOADED DATA TAIL\n", data.tail(5))

    # visualise correlation table
    if visualise_corr:
        DataVisualiser.visualise_correlation_table(data)

    # visualise tickers
    for v_ticker in visualise_tickers:
        # get columns with ticker if ticker in data
        if v_ticker in tickers:
            v_ticker = v_ticker.lower()
            ohlcv_columns = \
                [f"{v_ticker}_open", f"{v_ticker}_high", f"{v_ticker}_low", f"{v_ticker}_close", f"{v_ticker}_volume"]
            visualise_df = data[[_c for _c in ohlcv_columns]]
            DataVisualiser.visualise_df_ohlc(visualise_df, v_ticker)

    return data


def classify_data(data):

    # mandatory params
    target_col = configs_json["classify_data"]["target_col"]
    t_in_future = configs_json["classify_data"]["t_in_future"]

    # optional params
    delta = configs_json["classify_data"]["delta"] if "delta" in configs_json["classify_data"] else None
    span = configs_json["classify_data"]["span"] if "span" in configs_json["classify_data"] else None
    visualise = bool(configs_json["classify_data"]["visualise"]) if "visualise" in configs_json["classify_data"] else False

    classifier = DataClassifier()
    # calculate label column and assign it to the original data
    label_col = classifier.classify(data, target_col, t_in_future, delta, span)
    data = data.assign(label=label_col)
    # keep the rows where label is not NA
    data = data[data["label"].notna()]

    print("CLASSIFIED DATA HEAD\n", data.head(5))
    print("CLASSIFIED DATA TAIL\n", data.tail(5))

    if visualise:
        DataVisualiser.visualise_clf(data, target_col)

    return data


def process_data(data, classify):

    # mandatory params
    split_rate = configs_json["process_data"]["split_rate"]
    nn_training_params = configs_json["nn_model"]["training"]

    # optional params
    target_col = configs_json["process_data"]["target_col"] if "target_col" in configs_json["process_data"] else None
    columns = configs_json["process_data"]["columns"] if "columns" in configs_json["process_data"] else None
    validation_rate = configs_json["process_data"]["validation_rate"] if "validation_rate" in configs_json["process_data"] else None
    seq_len = configs_json["process_data"]["seq_len"] if "seq_len" in configs_json["process_data"] else None
    shuffle = configs_json["process_data"]["shuffle"] if "shuffle" in configs_json["process_data"] else None
    processing_type = configs_json["process_data"]["processing_type"] if "processing_type" in configs_json["process_data"] else None
    use_train_gen = bool(nn_training_params["use_train_gen"]) if "use_train_gen" in nn_training_params else None

    data_process = DataProcess(data, split_rate, classify, target_col, columns, validation_rate)

    train_gen, x_train, y_train, x_test, y_test, x_validation, y_validation, input_features = \
        None, None, None, None, None, None, None, None

    # get generator reference or full training data in memory
    if use_train_gen:
        train_gen = data_process.generate_train_batch
    else:
        x_train, y_train = data_process.get_train_data(seq_len, shuffle, processing_type)

    # get validation data
    if validation_rate:
        x_validation, y_validation = data_process.get_validation_data(seq_len, shuffle, processing_type)

    # get test data
    x_test, y_test = data_process.get_test_data(seq_len, processing_type)

    # get nr of features in training data (required for nn_model)
    input_features = data_process.data_train.shape[1] - 1

    return train_gen, x_train, y_train, x_test, y_test, x_validation, y_validation, input_features


def train_nn_model(train_gen=None, x_train=None, y_train=None, x_validation=None, y_validation=None, input_seq=None, input_features=None):

    nn_model = NNModel()
    load_model_filename = configs_json["nn_model"]["load_model_filename"]

    if load_model_filename and len(load_model_filename) > 0:
        # load model
        file_path = path.join(path.dirname(path.abspath(__file__)), "saved_models", load_model_filename)
        nn_model.load_model(file_path)
    else:
        # train model
        nn_model.build_model(configs_json, input_seq, input_features)

        nn_model_params = configs_json["nn_model"]["training"]
        epochs = nn_model_params["epochs"]
        batch_size = nn_model_params["batch_size"]

        if train_gen:
            # use training generator, do not load all data into memory
            nn_model.train_generator(train_gen, epochs, batch_size, x_validation, y_validation)
        elif x_train is not None and y_train is not None:
            classify = configs_json["classify_data"]["classify"]

            if classify:
                # transform y_train values in one-hot encoders
                y_train = to_categorical(y_train)

                if x_validation is not None and y_validation is not None:
                    # transform y_validation values in one-hot encoders
                    y_validation = to_categorical(y_validation)

            nn_model.train(x_train, y_train, epochs, batch_size, x_validation, y_validation)

    return nn_model


def train_clf_model(x_train, y_train, x_validation=None, y_validation=None):

    clf_model = CLFModel()

    model_type = configs_json["clf_model"]["type"]
    load_model_filename = configs_json["clf_model"]["load_model_filename"]

    if load_model_filename and len(load_model_filename) > 0:
        # load clf model
        file_path = path.join("saved_models", load_model_filename)
        clf_model.load_model(file_path)

    else:
        save_fname = path.join("saved_models", '%s-%s.pkl'
                               % (str(model_type), dt.datetime.now().strftime('%d-%m-%Y__%H-%M-%S')))
        # build model
        clf_model.build_model(configs_json)
        # train and save model
        clf_model.train(x_train, y_train, x_validation, y_validation)
        clf_model.save_model(save_fname)

    return clf_model


def run_pipeline():
    # fetch data
    fetch_data_from_api = bool(configs_json["fetch_data"]["fetch"])

    if fetch_data_from_api:
        has_data_fetch = fetch_data()
        fetch_msg = "Data has been successfully fetched!" if has_data_fetch else "Data has not been fetched"
        print(fetch_msg)

    # load data
    data = load_data()

    # classify data
    classify = bool(configs_json["classify_data"]["classify"])

    if classify:
        data = classify_data(data)
        labels_count = Counter(data["label"].dropna().values.astype(int))
        print(labels_count)

    # process data
    train_gen, x_train, y_train, x_test, y_test, x_validation, y_validation, input_features = process_data(data, classify)

    model_type = configs_json["model_type"]

    # train model; make predictions; test; produce visualisations

    if model_type == "nn_model":
        input_seq = configs_json["process_data"]["seq_len"] - 1
        # train nn model
        trained_nn_model = \
            train_nn_model(train_gen, x_train, y_train, x_validation, y_validation, input_seq, input_features)

        if not classify:
            # make point by point and sequence-based predictions
            predictions = trained_nn_model.predict_point_by_point(x_test)
            predictions_seq = trained_nn_model.predict_sequences_multiple(
                x_test,
                configs_json["process_data"]["seq_len"],
                configs_json["process_data"]["seq_len"])
            # visualise results
            DataVisualiser.plot_results_multiple(predictions_seq, y_test, configs_json["process_data"]["seq_len"])
            DataVisualiser.plot_results(predictions, y_test)
        else:

            y_test_encoded = to_categorical(y_test)
            score = trained_nn_model.model.evaluate(x_test, y_test_encoded, verbose=1)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    elif model_type == "clf_model":
        # reshape X data
        if len(x_train.shape) > 2:
            x_train = flatten_3d_data(x_train)

        if len(x_test.shape) > 2:
            x_test = flatten_3d_data(x_test)

        if x_validation is not None and len(x_validation.shape) > 2:
            x_validation = flatten_3d_data(x_validation)

        # reshape y data
        y_train = y_train.reshape((y_train.shape[0], ))
        y_test = y_test.reshape((y_test.shape[0], ))

        if y_validation is not None:
            y_validation = y_validation.reshape((y_validation.shape[0],))

        trained_clf_model = train_clf_model(x_train, y_train, x_validation, y_validation)
        predictions = trained_clf_model.predict_point_by_point(x_test)

        # calculate accuracy
        clf_acc = 0

        for idx, prediction in enumerate(predictions):
            if int(prediction) == y_test[idx]:
                clf_acc += 1

        clf_acc = (clf_acc / y_test.shape[0]) * 100
        print(f"The accuracy of the [CLFModel] is: {clf_acc}%")
        # calculate classification report
        print(classification_report(y_test, predictions))


if __name__ == "__main__":
    run_pipeline()













