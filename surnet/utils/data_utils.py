import numpy as np


def min_max_normalize(x, strip_dim=False):
    if strip_dim:
        x -= x.min(0)[0]
        mx = x.max(0)[0]
    else:
        x -= x.min(0)
        mx = x.max(0)
    if (mx == 0).any():  # avoid division by zero
        mx[mx == 0] = 1
        x = 2 * (x / mx) - 1
        x[:, mx == 1] = 0.5
    else:
        x = 2 * (x / mx) - 1
    return x


# transforms data between -0.5 and 0.5
def range_normalize(x, low=-5, high=5):
    size = high - low
    center = (low + high) / 2
    return (x + center) / size


def validation_split(x, y, ratio=0.1, min_train=2):
    samples = len(x)
    assert len(y) == samples

    mask = np.arange(samples)
    np.random.shuffle(mask)  # TODO smart sampling

    x = x[mask]
    y = y[mask]

    m = min(samples, min_train)
    train = max(m, int((1 - ratio) * samples))
    val = samples - train

    x_train, x_val = x[val:], x[:val]
    y_train, y_val = y[val:], y[:val]

    return x_train, x_val, y_train, y_val
