from torch.nn import functional


class UpSample:
    """
    `nn.Upsample` being deprecated, we re-wrote a class using the functional `interpolate`.

    This enables to define the up-sampler once only.
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        self.args = {
            'size': size,
            'scale_factor': scale_factor,
            'mode': mode
        }

    def __call__(self, x):
        """
        Interpolates the tensor.

        Args:
            x (torch.tensor):

        Returns:
            torch.Tensor: The interpolated tensor.
        """

        return functional.interpolate(x, **self.args)
