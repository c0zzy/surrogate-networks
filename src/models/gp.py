import torch

import gpytorch

import numpy as np
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

from src.models.surrogate import Surrogate


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(
            nu=5 / 2,
            lengthscale=1.,
        ))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPSurrogate(Surrogate):
    def __init__(self, **hps):
        super().__init__()

        defaults = {
            'gp_noise': 1e-4,
            'opt_iters': 200,
            'lr': 0.05,
        }
        defaults.update(hps)
        self.hps = defaults

        self.likelihood = GaussianLikelihood()
        self.likelihood.initialize(noise=self.hps['gp_noise'])

    def fit(self, x, y, iters=None):
        if not iters:
            iters = self.hps['opt_iters']

        train_x = torch.from_numpy(x.astype(np.float32))
        train_y = torch.from_numpy(y.astype(np.float32))

        self.regressor = GPRegressionModel(train_x, train_y, self.likelihood)

        # set train mode
        self.regressor.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.hps['lr'])

        mll = ExactMarginalLogLikelihood(self.likelihood, self.regressor)

        for i in range(iters):
            optimizer.zero_grad()
            output = self.regressor(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return self.regressor

    def predict(self, x, var=False):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_x = torch.from_numpy(x.astype(np.float32))

            # set eval mode
            self.regressor.eval()
            self.likelihood.eval()

            prediction = self.likelihood(self.regressor(pred_x))
            if var:
                return prediction.mean.numpy(), prediction.variance.numpy()
            return prediction.mean.numpy()
