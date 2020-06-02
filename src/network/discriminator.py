from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, Tuple

import numpy
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input
from keras.models import Model, Sequential
from keras.optimizers import Optimizer


class Param(Enum):
    OUT_SHAPE = "disc_out_shape"
    IN_SHAPE = "disc_in_shape"
    N_LAYERS = "disc_num_layers"
    LAYER_MULTIPLIER = "disc_layer_multiplier"
    BATCH_NORM_MOMENTUM = "disc_bn_momentum"
    LEAKY_RELU_ALPHA = "disc_leaky_relu_alpha"
    DROPOUT = "disc_dropout"
    OPTIMIZER_TYPE = "disc_opt_type"
    OPTIMIZER_PARAMS = "disc_opt_params"


class Discriminator(ABC, Model):

    CONDITIONAL: ClassVar[bool] = NotImplemented
    INFINITE: ClassVar[bool] = NotImplemented

    def __init__(self, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]):
        self.in_shape = in_shape
        self.out_shape = out_shape

    @abstractmethod
    def create_param_dict(self) -> Dict[str, Any]:
        pass


class SimpleMlpDisc(Discriminator):

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

        Discriminator.__init__(self, in_shape, (1,))

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.leaky_relu_alpha = leaky_relu_alpha

        x = Input(in_shape)
        y = Flatten()(x)

        for layer in range(num_layers):
            y = Dense(
                round(numpy.prod(in_shape) * ((num_layers - layer) / num_layers))
            )(y)
            y = LeakyReLU(alpha=leaky_relu_alpha)(y)
            y = Dropout(dropout)(y)

        y = Dense(1, activation="sigmoid")(y)

        Model.__init__(self, inputs=x, outputs=y, name=name or self.__class__.__name__)

        self.compile(
            loss=["binary_crossentropy"], optimizer=optimizer, metrics=["accuracy"]
        )


    def create_param_dict(self) -> Dict[str, Any]:
        return {
            Param.OUT_SHAPE.value: self.out_shape,
            Param.IN_SHAPE.value: self.in_shape,
            Param.N_LAYERS.value: self.num_layers,
            Param.LAYER_MULTIPLIER.value: self.layer_multiplier,
            Param.BATCH_NORM_MOMENTUM.value: self.bn_momentum,
            Param.LEAKY_RELU_ALPHA.value: self.leaky_relu_alpha,
            Param.DROPOUT.value: self.dropout,
            Param.OPTIMIZER_TYPE.value: self.optimizer.__class__.__name__,
            Param.OPTIMIZER_PARAMS.value: self.optimizer.get_config(),
        }
