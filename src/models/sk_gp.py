from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.models.surrogate import Surrogate


class SkGPSurrogate(Surrogate):
    def __init__(self):
        super().__init__()

        self.regressor = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=1),
            alpha=0
        )

    def fit(self, x, y):
        self.regressor.fit(x, y)
        return self.regressor

    def predict(self, x):
        return self.regressor.predict(x)
