import numpy as np
import torch


def upside_down_inversion(x):
    # x = time series, dimension = n_values
    # negate the values of x
    x_flip = (-1.0 * x).copy()
    return x_flip


def adding_partial_noise(x, second=10, duration=2, num_samples_per_second=125):
    # x = time series, dimension = n_values
    # replace the signal belonging to interval [second, second+duration] by
    # a noise that matches the mean and variance of the dropped samples.
    begin = second * num_samples_per_second
    end = (second + duration) * num_samples_per_second
    section = x[begin:end]
    delta_x = np.std(section)
    mean_x = np.mean(section)
    noise = np.random.normal(mean_x, delta_x, end - begin)
    x_noise = x.copy()
    x_noise[begin:end] = noise
    return x_noise


def shift_series(x, shift=10):
    # shift the time series by shift steps
    shifted_x = np.concatenate((x[shift:], x[:shift] + x[-1] - x[0]))
    return shifted_x


def transform_tensor_sample(x):
    # first, random flip
    if np.random.random() > 0.5:
        x = -1.0 * x
    # shift the series by 1 to 25 steps
    if np.random.random() > 0.5:
        shift = np.random.randint(1, 26)
        x = torch.cat(
            (x[:, shift:], x[:, :shift] + x[:, x.size(1) - 1] - x[:, 0]), dim=1
        )
    # add partial gaussian noise 50% of the time
    if np.random.random() > 0.5:

        second = np.random.randint(0, 29)
        duration = np.random.randint(1, 3)

        num_samples_per_second = 125
        begin = second * num_samples_per_second
        end = (second + duration) * num_samples_per_second

        section = x[:, begin:end]
        delta_x = torch.std(section, dim=1, keepdim=True)
        mean_x = torch.mean(section, dim=1, keepdim=True)

        noise = torch.randn(x.size(0), end - begin).to(x.device)
        noise = delta_x * noise + mean_x

        x_noise = torch.cat((x[:, :begin], noise, x[:, end:]), dim=1)

        x = x_noise

    return x


def noisy_sample(x):
    # return transform_tensor_sample(x)
    return torch.cat(
        [transform_tensor_sample(x[i:i + 1]) for i in range(x.size(0))], 0
    )

