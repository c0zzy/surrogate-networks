from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from src.surrogate import Surrogate


class SkGPSurrogate(Surrogate):
    def __init__(self):
        super().__init__()
        kernel = 1.0 * RBF(1.0)
        self.regressor = GaussianProcessRegressor(kernel=kernel)

    def fit(self, x, y):
        self.regressor.fit(x, y)
        return self.regressor

    def predict(self, x):
        return self.regressor.predict(x)
