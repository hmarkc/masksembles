import torch
from torch import nn

from . import common

class Masksembles2D(nn.Module):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`torch.nn.Dropout2d`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N, C, H, W) (same shape as input)

    Examples:

    >>> m = Masksembles2D(16, 4, 2.0)
    >>> input = torch.ones([4, 16, 28, 28])
    >>> output = m(input)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale
        self.cnt = 0

        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).float()
        if torch.cuda.is_available():
            self.masks = self.masks.cuda()

    def forward(self, inputs):
        batch = inputs.shape[0]
        if self.training:
            if batch % self.n != 0:
                raise ValueError('Batch size must be divisible by n, got batch {} and n {}'.format(batch, self.n))
            x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
            x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
            x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        else: 
            x = inputs * self.masks[self.cnt][None].unsqueeze(1).unsqueeze(-1).unsqueeze(-1) 
            # print("2D Sampling... Using mask: ", self.cnt)
            self.cnt = (self.cnt + 1) % self.n     
        return x.squeeze(0).float()
    
    def extra_repr(self):
        return 'scale={}, n={}'.format(
            self.scale, self.n
        )


class Masksembles1D(nn.Module):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`torch.nn.Dropout`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(16, 4, 2.0)
    >>> input = torch.ones([4, 16])
    >>> output = m(input)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):

        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale
        self.cnt = 0

        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).float() 
        if torch.cuda.is_available():
            self.masks = self.masks.cuda()

    def forward(self, inputs):
        batch = inputs.shape[0]
        if self.training:
            if batch % self.n != 0:
                raise ValueError('Batch size must be divisible by n, got batch {} and n {}'.format(batch, self.n))
            x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
            x = torch.cat(x, dim=1).permute([1, 0, 2])
            x = x * self.masks.unsqueeze(1)
            x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        else:
            x = inputs * self.masks[self.cnt][None].unsqueeze(1)
            # print("1D Sampling... Using mask: ", self.cnt)
            self.cnt = (self.cnt + 1) % self.n
        return x.squeeze(0)
    
    def extra_repr(self):
        return 'scale={}, n={}'.format(
            self.scale, self.n
        )
