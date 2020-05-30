from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

import numpy
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU, Reshape
from keras.models import Sequential


class Generator(ABC, Sequential):

    CONDITIONAL: ClassVar[bool] = NotImplemented
    INFINITE: ClassVar[bool] = NotImplemented

    def __init__(
        self, latent_size: int, out_shape: Tuple[int, ...], name: str = None,
    ):
        self.latent_size = latent_size
        self.out_shape = out_shape

        Sequential.__init__(self, name=name if name else self.__class__.__name__)


class NonConditionalMlpGen(Generator):

    CONDITIONAL = False
    INFINITE = False

    def __init__(
        self,
        latent_size: int,
        out_shape: Tuple[int, ...],
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        leaky_relu_alpha: float,
        name: str = None,
    ):

        Generator.__init__(self, latent_size, out_shape, name)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.leaky_relu_alpha = leaky_relu_alpha

        self.add(Dense(latent_size, input_shape=(latent_size,)))
        self.add(BatchNormalization(momentum=bn_momentum))
        self.add(LeakyReLU(alpha=leaky_relu_alpha))

        for layer in range(2, num_layers):
            self.add(
                Dense(
                    round(
                        latent_size
                        + (
                            abs(latent_size - numpy.prod(out_shape))
                            * (layer / (num_layers - 1))
                        )
                    )
                )
            )
            self.add(BatchNormalization(momentum=bn_momentum))
            self.add(LeakyReLU(alpha=leaky_relu_alpha))
            self.add(Dropout(0.5))
        self.add(Dense(numpy.prod(out_shape), activation="tanh"))
        self.add(Reshape(out_shape))
