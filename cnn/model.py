"""Module for machine learning model classes tailored to active noise cancellation"""

import torch
from torch import nn

DEFAULT_NB_CHANNELS = 2
DEFAULT_KERNEL_H = [11, 9, 7, 5]
DEFULT_NB_FILTERS = [64, 32, 16, 2]


class CNN4ANC(nn.Module):
    """Convolutional Neural Network (CNN) for Active Noise Cancellation (ANC).

    This class implements a customizable CNN model designed to predict and cancel out noise in
    signals. The model is constructed with a series of convolutional layers, each configured with
    specific filter counts and convolution window sizes. These layers collaboratively process the
    input Electromagnetic Interference (EMI) signals, learning to identify noise patterns. The
    model predicts the noise component within the MRI signal, which can subsequently be subtracted
    from the original MRI signal to effectively cancel out the noise.


    Attributes:
        nb_channels : number of input EMI channels.
        kernel_h : convolution window for each layer along the height axis (i.e. length of signal).
        nb_filters : number of filters (output channels) for each convolutional layer.
        nb_layers : number of convolutional layers in the model.
        model : sequential container holding the layers of the CNN model.
    """

    def __init__(
        self,
        nb_channels: int = DEFAULT_NB_CHANNELS,
        kernel_h: list[int] = DEFAULT_KERNEL_H,
        nb_filters: list[int] = DEFULT_NB_FILTERS,
    ):
        """Initialize CNN4ANC model with given parameters.

        Args:
            nb_channels : number of input EMI channels.
            kernel_h : kernel sizes for each layer along the height axis.
            nb_filters : number of filters for each layer.
        """
        super(CNN4ANC, self).__init__()

        self.nb_channels = nb_channels
        self.kernel_h = kernel_h
        self.nb_filters = nb_filters
        self.nb_layers = len(kernel_h)

        layers = []
        for i in range(self.nb_layers):
            if i == 0:
                in_channels = 2
                kernel = (self.kernel_h[i], self.nb_channels)
            else:
                in_channels = self.nb_filters[i - 1]
                kernel = (self.kernel_h[i], 1)
            out_channels = self.nb_filters[i]
            pad = (kernel[0] // 2, 0)
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=pad)
            )
            layers.append(nn.BatchNorm2d(self.nb_filters[i]))
            if i != (self.nb_layers - 1):
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model to predict noise.

        Args:
            input : input tensor with shape (batch_size, 2, nb_samples_per_shot, nb_channels).

        Returns:
            output tensor with predicted noise.
        """
        return self.model(input)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=(11,2),stride=1,padding=(5,0)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64,32,kernel_size=(9,1),stride=1,padding=(4,0)),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32,16,kernel_size=(7,1),stride=1,padding=(3,0)),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16,2,kernel_size=(5,1),stride=(1,1),padding=(2,0)),
            nn.BatchNorm2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        return x