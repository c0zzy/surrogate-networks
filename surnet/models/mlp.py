import warnings

from sklearn.neural_network import MLPRegressor

from ..models.surrogate import Surrogate


class MLPSurrogate(Surrogate):
    def __init__(self):
        super().__init__()
        self.regressor = MLPRegressor(random_state=1, max_iter=500)

    def fit(self, x, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regressor.fit(x, y)
        return self.regressor

    def predict(self, x):
        return self.regressor.predict(x)
