from abc import ABC
from typing import Tuple

import numpy
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Optimizer


class Discriminator(ABC, Sequential):
    def __init__(
        self, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], name: str = None
    ):
        self.in_shape = in_shape
        self.out_shape = out_shape

        Sequential.__init__(self, name=name if name else self.__class__.__name__)


class MlpDisc(Discriminator):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        num_classes: int,
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        dropout: float,
        leaky_relu_alpha: float,
        optimizer: Optimizer,
        name: str = None,
    ):

        Discriminator.__init__(self, in_shape, (num_classes,), name)

        self.in_shape = in_shape
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.leaky_relu_alpha = leaky_relu_alpha

        Discriminator.add(self, Flatten(input_shape=in_shape))

        for layer in range(num_layers):
            Discriminator.add(
                self,
                [
                    Dense(
                        numpy.prod(in_shape)
                        / (layer_multiplier * (num_layers - layer - 1))
                    ),
                    LeakyReLU(alpha=leaky_relu_alpha),
                    Dropout(dropout),
                ],
            )

        Discriminator.add(self, Dense(num_classes, activation="softmax"))

        Sequential.compile(
            self,
            loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
            optimizer=optimizer,
        )
