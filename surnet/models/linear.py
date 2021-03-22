from sklearn.linear_model import LinearRegression

from surnet.models.surrogate import Surrogate


class LinearSurrogate(Surrogate):
    def __init__(self):
        super().__init__()

        self.regressor = LinearRegression(normalize=True)

    def fit(self, x, y):
        self.regressor.fit(x, y)
        return self.regressor

    def predict(self, x):
        return self.regressor.predict(x)
