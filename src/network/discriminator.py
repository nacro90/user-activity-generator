from abc import ABC
from typing import ClassVar, Tuple

import numpy
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Optimizer


class Discriminator(ABC, Sequential):

    CONDITIONAL: ClassVar[bool] = NotImplemented
    INFINITE: ClassVar[bool] = NotImplemented

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        infinite: bool,
        name: str = None,
    ):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.infinite = infinite

        Sequential.__init__(self, name=name if name else self.__class__.__name__)


class NonConditionalMlpDisc(Discriminator):

    CONDITIONAL = False
    INFINITE = True

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        dropout: float,
        leaky_relu_alpha: float,
        optimizer: Optimizer,
        name: str = None,
    ):

        Discriminator.__init__(self, in_shape, (1,), False, name)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.leaky_relu_alpha = leaky_relu_alpha

        self.add(Flatten(input_shape=in_shape))

        for layer in range(num_layers):
            self.add(
                Dense(round(numpy.prod(in_shape) * ((num_layers - layer) / num_layers)))
            )
            self.add(LeakyReLU(alpha=leaky_relu_alpha))
            self.add(Dropout(dropout))

        self.add(Dense(1, activation="sigmoid"))
        self.compile(
            loss=["binary_crossentropy"], optimizer=optimizer, metrics=["accuracy"]
        )
