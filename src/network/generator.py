from abc import ABC
from typing import Tuple

import numpy
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU, Reshape
from keras.models import Sequential


class Generator(ABC, Sequential):
    def __init__(self, latent_size: int, out_shape: Tuple[int, ...], name: str = None):
        self.latent_size = latent_size
        self.out_shape = out_shape

        Sequential.__init__(self, name=name if name else self.__class__.__name__)


class MlpGen(Generator):
    def __init__(
        self,
        latent_size: int,
        out_shape: Tuple[int, ...],
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        dropout: float,
        leaky_relu_alpha: float,
        name: str = None,
    ):

        Generator.__init__(self, latent_size, out_shape, name)

        self.latent_size = latent_size
        self.out_shape = out_shape
        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.leaky_relu_alpha = leaky_relu_alpha

        for layer in range(num_layers):
            Generator.add(
                self,
                [
                    Dense(
                        layer * layer_multiplier * latent_size,
                        input_dim=self.latent_size,
                    ),
                    BatchNormalization(momentum=bn_momentum),
                    LeakyReLU(alpha=leaky_relu_alpha),
                    Dropout(dropout),
                ],
            )
        Generator.add(
            self, [Dense(numpy.prod(out_shape), activation="tanh"), Reshape(out_shape)]
        )
