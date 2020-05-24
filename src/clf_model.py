"""
============IMPORTS============
"""
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import numpy as np
import pickle

class CLFModel:
    """
    Provides abstractions for:
        1. builds classification model, based on config file
        2. trains classification model using X and y training data
        3. predicts y point-by-point, using real past data
        4. evaluates model accuracy using X and y test data

    The class momentarily only supports the following classification algorithms:
        a) Logistic Regression (lr)
        b) XGB Classifier (xgb)
        c) Support Vector Machines (svm)
    """
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_params = None

    def build_model(self, configs):
        # get the required model type
        supported_models = ["svm", "xgb", "lr"]
        model_type = configs["clf_model"]["type"]

        # check if the required model is supported
        if model_type in supported_models:
            # check for params
            model_params = configs["clf_model"]["model_params"] if "model_params" in configs["clf_model"] else None

            if model_type == "svm":
                self.model = SVC(**model_params) if model_params else SVC()
            elif model_type == "xgb":
                self.model = XGBClassifier(**model_params) if model_params else XGBClassifier()
            elif model_type == "lr":
                self.model = LogisticRegression(**model_params) if model_params else LogisticRegression()

            self.model_type = model_type
            self.model_params = model_params
        else:
            raise TypeError("Classifier Model Type is not supported! Please try one of the following models:"
                            "'SVM' | 'XGBClassifier' | 'Logistic Regression'")

        print("[CLFModel] Model Compiled: %s-%s" % (model_type, model_params))

    def train(self, x_train, y_train, x_validation=None, y_validation=None):

        print("[CLFModel] Training Started")

        self.model.fit(x_train, y_train)

        if x_validation is not None and y_validation is not None:
            print("[CLFModel] Accuracy score for provided validation data:")
            score = self.evaluate_model(x_validation, y_validation)
            print(f"The score of the current {self.model_type} model is: {score}")

    def predict_point_by_point(self, data):
        print("[CLFModel] Prediction Process Started")

        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size, ))

        return predicted

    def evaluate_model(self, x, y):
        score = self.model.score(x, y)
        return score

    def save_model(self, save_fname):
        print("[CLFModel] Saving model...")

        print(save_fname)
        with open(save_fname, "wb") as file:
            pickle.dump(self.model, file)

        print(f"[CLFModel] The model has been saved as: {save_fname}")

    def load_model(self, load_fname):
        print("[CLFModel] Loading model...")

        with open(load_fname, "rb") as file:
            self.model = pickle.load(file)

