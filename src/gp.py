import torch
import gpytorch

import numpy as np

from src.surrogate import Surrogate


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        nu = 5 / 2
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=nu)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSurrogate(Surrogate):
    def __init__(self):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def fit(self, x, y, iters=50):
        train_x = torch.from_numpy(x.astype(np.float32))
        train_y = torch.from_numpy(y.astype(np.float32))

        self.regressor = GPRegressionModel(train_x, train_y, self.likelihood)

        # set train mode
        self.regressor.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.regressor)

        for i in range(iters):
            optimizer.zero_grad()
            output = self.regressor(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return self.regressor

    def predict(self, x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_x = torch.from_numpy(x.astype(np.float32))

            # set eval mode
            self.regressor.eval()
            self.likelihood.eval()

            prediction = self.likelihood(self.regressor(pred_x))
            return prediction.mean.numpy(), prediction.variance.numpy()
