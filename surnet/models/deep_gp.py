import itertools

import gpytorch
import torch

from models.gp import GPSurrogate, GPRegressionModel


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, dim, arch, activation=torch.nn.Sigmoid):
        super(FeatureExtractor, self).__init__()

        if isinstance(arch, str):
            arch = self.calculate_arch(arch, dim)

        self.add_module('linear1', torch.nn.Linear(dim, arch[0]))
        for i in range(len(arch) - 1):
            now, nxt = arch[i:i + 2]
            self.add_module(f'activation{i + 1}', activation())
            self.add_module(f'linear{i + 2}', torch.nn.Linear(now, nxt))

    @staticmethod
    def calculate_arch(arch, dim, k=15):
        outsizes = {
            2: 2,
            3: 3,
            5: 3,
            10: 5,
            20: 5
        }
        out = outsizes[dim]
        if arch == 'low':
            return [out, out]
        elif arch == 'high':
            gp_tp = 12  # TODO spectral mixture tp?
            max_tp = (k * dim) - gp_tp
            hid = int((max_tp - out) / (dim + out - 1))
            return [hid, out]


class DeepGPRegressionModel(GPRegressionModel):
    def __init__(self, x_train, y_train, likelihood, hps):
        super(DeepGPRegressionModel, self).__init__(x_train, y_train, likelihood, hps)

        dim = x_train.size()[1]
        arch = hps['model']['arch']
        # activation = self._select_activation(hps['model']['activation'])  # TODO
        self.feature_extractor = FeatureExtractor(dim, arch)

    def forward(self, x):
        # We're first putting our data through an MLP (feature extractor)

        x_proj = self.feature_extractor(x)

        # TODO examine normalization (seems like bs with our data)
        # x_proj = min_max_normalize(x_proj)

        x_mean = self.mean_module(x_proj)
        x_covar = self.covar_module(x_proj)
        return gpytorch.distributions.MultivariateNormal(x_mean, x_covar)

    def trainable_params(self):
        return itertools.chain(
            self.feature_extractor.parameters(),
            self.covar_module.parameters(),
            self.mean_module.parameters(),
            self.likelihood.parameters()
        )


class DeepGPSurrogate(GPSurrogate):
    def __init__(self, hps, **kwargs):
        super().__init__(hps, **kwargs)
        self.regressor_class = DeepGPRegressionModel
