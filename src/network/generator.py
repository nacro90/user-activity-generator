from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, Tuple

import numpy
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU, Reshape
from keras.models import Sequential


class Param(Enum):
    LATENT_SIZE = "gen_latent_size"
    OUT_SHAPE = "gen_out_shape"
    N_LAYERS = "gen_num_layers"
    LAYER_MULTIPLIER = "gen_layer_multiplier"
    BATCH_NORM_MOMENTUM = "gen_bn_momentum"
    LEAKY_RELU_ALPHA = "gen_leaky_relu_alpha"
    DROPOUT = "gen_dropout"


class Generator(ABC, Sequential):

    CONDITIONAL: ClassVar[bool] = NotImplemented
    INFINITE: ClassVar[bool] = NotImplemented

    def __init__(
        self, latent_size: int, out_shape: Tuple[int, ...], name: str = None,
    ):
        self.latent_size = latent_size
        self.out_shape = out_shape

        Sequential.__init__(self, name=name or self.__class__.__name__)

    @abstractmethod
    def create_param_dict(self) -> Dict[str, Any]:
        pass


class SimpleMlpGen(Generator):

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
        dropout: float,
        name: str = None,
    ):

        Generator.__init__(self, latent_size, out_shape, name)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout

        self.add(Dense(latent_size, input_shape=(latent_size,)))
        self.add(BatchNormalization(momentum=bn_momentum))
        self.add(LeakyReLU(alpha=leaky_relu_alpha))
        self.add(Dropout(dropout))

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
            self.add(Dropout(dropout))
        self.add(Dense(numpy.prod(out_shape), activation="tanh"))
        self.add(Reshape(out_shape))

    def create_param_dict(self) -> Dict[str, Any]:
        return {
            Param.LATENT_SIZE.value: self.latent_size,
            Param.OUT_SHAPE.value: self.out_shape,
            Param.N_LAYERS.value: self.num_layers,
            Param.LAYER_MULTIPLIER.value: self.layer_multiplier,
            Param.BATCH_NORM_MOMENTUM.value: self.bn_momentum,
            Param.LEAKY_RELU_ALPHA.value: self.leaky_relu_alpha,
            Param.DROPOUT.value: self.dropout,
        }
