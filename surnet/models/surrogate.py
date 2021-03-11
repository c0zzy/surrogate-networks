import numpy as np
from utils.logger import log


class Surrogate:
    def __init__(self, log=log, **kwargs):
        self.regressor = None
        self.log = log

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        diff_pred = np.abs(y_true - y_pred)
        return tuple([f(diff_pred) for f in [np.min, np.max, np.average]])
