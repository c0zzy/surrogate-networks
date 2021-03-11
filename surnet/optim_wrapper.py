from torch.optim import Adam, LBFGS


class OptimizerWrapper:
    def __init__(self, model, loss, x, y):
        self.model = model
        self.loss = loss
        self.x = x
        self.y = y
        self.optimizer = None

    def closure(self):
        self.optimizer.zero_grad()
        predicted = self.model(self.x)
        loss = - self.loss(predicted, self.y)
        loss.backward()
        return loss

    def step(self):
        loss = self.closure()
        self.optimizer.step()
        return loss


class AdamWrapper(OptimizerWrapper):
    def __init__(self, model, loss, x, y, lr=1e-3):
        super(AdamWrapper, self).__init__(model, loss, x, y)

        self.optimizer = Adam(model.trainable_params(), lr=lr)


class LBFGSWrapper(OptimizerWrapper):
    def __init__(self, model, loss, x, y, lr=1e-3):
        super(LBFGSWrapper, self).__init__(model, loss, x, y)

        self.optimizer = LBFGS(model.trainable_params(), lr=lr)

    def step(self):
        self.optimizer.step(self.closure)
        return self.closure()
