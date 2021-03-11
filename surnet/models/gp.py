import copy

import torch

import gpytorch

import numpy as np
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean

from losses import LossL2
from models.kernels import init_kernel
from models.surrogate import Surrogate
from optim_wrapper import AdamWrapper, LBFGSWrapper
from utils.data_utils import range_normalize, validation_split


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, hps):
        super(GPRegressionModel, self).__init__(x_train, y_train, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = init_kernel(dim=x_train.shape[1], **hps['model']['kernel'])

    def forward(self, x):
        x_mean = self.mean_module(x)
        x_covar = self.covar_module(x)

        problem = torch.isnan(x_covar.evaluate()).any()  # TODO remove
        if problem:
            self.covar_module(x)

        return MultivariateNormal(x_mean, x_covar)

    def trainable_params(self):
        return self.parameters()
        # return [p for n, p, in self.named_parameters() if 'lengthscale' not in n]

    def count_trainable_params(self):
        return sum([np.prod(p.size(), dtype=int) for p in self.parameters()])


class GPSurrogate(Surrogate):
    def __init__(self, hps, double_precision=True, **kwargs):
        super().__init__(**kwargs)

        self.hps = hps
        self.likelihood = GaussianLikelihood()
        self.likelihood.initialize(noise=self.hps['model']['noise'])

        self.regressor_class = GPRegressionModel

        self.np_dtype = np.float64 if double_precision else np.float32
        if double_precision:
            torch.set_default_dtype(torch.double)

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        self.regressor = None

    class train_switch:  # maybe TODO
        def __init__(self, surrogate):
            pass

        def __enter__(self):
            pass

        def __exit__(self):
            pass

    def set_eval_mode(self):
        self.regressor.eval()
        self.likelihood.eval()

    def set_train_mode(self):
        self.regressor.train()
        self.likelihood.train()

    def init_optimizer(self, loss, x, y):
        optimizer_class = {
            'adam': AdamWrapper,
            'lbfgs': LBFGSWrapper
        }[self.hps['optimizer']['name']]

        return optimizer_class(self.regressor, loss, x, y, lr=self.hps['optimizer']['lr'])

    def validation_error(self, x_val, y_val, loss_fnc=LossL2.torch):
        training_mode = self.regressor.training
        self.set_eval_mode()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_val = self.likelihood(self.regressor(x_val)).mean
            err = loss_fnc(pred_val, y_val).detach().numpy()

        if training_mode:
            self.set_train_mode()
        return err

    def fit(self, x, y, iters=None, validate=True, reset=False, verbose=False, val_treshold=500):
        if not iters:
            iters = self.hps['optimizer']['iters']

        if self.regressor is None or reset:
            x = range_normalize(x)
            x_train_np, x_val_np, y_train_np, y_val_np = validation_split(x, y)

            self.x_train = torch.from_numpy(x_train_np.astype(self.np_dtype))
            self.y_train = torch.from_numpy(y_train_np.astype(self.np_dtype))
            self.x_val = torch.from_numpy(x_val_np.astype(self.np_dtype))
            self.y_val = torch.from_numpy(y_val_np.astype(self.np_dtype))

            self.regressor = self.regressor_class(self.x_train, self.y_train, self.likelihood, self.hps)

        params_cnt = self.regressor.count_trainable_params()

        if len(self.x_train) <= params_cnt:
            self.log(
                f"Not enough training data for current model, Samples: {len(self.x_train)} <= Params: {params_cnt}"
            )
            return None

        self.set_train_mode()
        mll = ExactMarginalLogLikelihood(self.likelihood, self.regressor)
        optimizer = self.init_optimizer(mll, self.x_train, self.y_train)

        min_val_err = np.inf
        worse = 0
        best_val_regressor = self.regressor
        best_i = 0
        for i in range(iters):
            optimizer.step()
            val_err = self.validation_error(self.x_val, self.y_val, loss_fnc=LossL2.torch)
            if val_err < min_val_err:
                min_val_err = val_err
                best_i = i
                worse = 0
                best_val_regressor = copy.deepcopy(self.regressor)
            else:
                worse += 1
                if worse > val_treshold:
                    break
        self.regressor = best_val_regressor

        ls = None
        try:
            ls = self.regressor.covar_module.base_kernel.lengthscale.detach().numpy()[0][0]
        except AttributeError:
            pass
        self.log(f"Best model from it: {best_i}, LS: {ls} ValErr: {min_val_err}")

        # self.log('lengthscale', self.regressor.covar_module.base_kernel.lengthscale.detach().numpy(), sep='\t')
        # self.log('lik noise', self.regressor.likelihood.noise.detach().numpy(), sep='\t')
        # self.log('const mean', self.regressor.mean_module.constant.detach().numpy(), sep='\t')
        # self.log('outputscale', self.regressor.covar_module.outputscale, sep='\t')

        return self.regressor

    def predict(self, x, var=False):
        x = range_normalize(x)

        training_mode = self.regressor.training
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_pred = torch.from_numpy(x.astype(self.np_dtype))
            self.set_eval_mode()
            prediction = self.likelihood(self.regressor(x_pred))
            if training_mode:
                self.set_train_mode()
            if var:
                return prediction.mean.numpy(), prediction.variance.numpy()
            return prediction.mean.numpy()
