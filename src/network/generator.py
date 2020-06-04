from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, Tuple

import numpy
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
    multiply,
)
from keras.models import Model, Sequential


class Param(Enum):
    LATENT_SIZE = "gen_latent_size"
    OUT_SHAPE = "gen_out_shape"
    N_LAYERS = "gen_num_layers"
    LAYER_MULTIPLIER = "gen_layer_multiplier"
    BATCH_NORM_MOMENTUM = "gen_bn_momentum"
    LEAKY_RELU_ALPHA = "gen_leaky_relu_alpha"
    DROPOUT = "gen_dropout"
    NUM_CLASSES = "gen_num_classes"


class Generator(ABC, Model):

    CONDITIONAL: ClassVar[bool] = NotImplemented
    INFINITE: ClassVar[bool] = NotImplemented

    def __init__(
        self, latent_size: int, out_shape: Tuple[int, ...], name: str = None,
    ):
        self.latent_size = latent_size
        self.out_shape = out_shape

    @abstractmethod
    def create_param_dict(self) -> Dict[str, Any]:
        pass

    def create_mlp_interim(
        self,
        x: Layer,
        out_shape: Tuple[int, ...],
        latent_size: int,
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        leaky_relu_alpha: float,
        dropout: float,
    ) -> Layer:

        y = Dense(latent_size)(x)
        y = BatchNormalization(momentum=bn_momentum)(y)
        y = LeakyReLU(alpha=leaky_relu_alpha)(y)
        y = Dropout(dropout)(y)
        for layer in range(2, num_layers):
            y = Dense(
                round(
                    latent_size
                    + (
                        abs(latent_size - numpy.prod(out_shape))
                        * (layer / (num_layers - 1))
                    )
                )
            )(y)
            y = BatchNormalization(momentum=bn_momentum)(y)
            y = LeakyReLU(alpha=leaky_relu_alpha)(y)
            y = Dropout(dropout)(y)
        return y


class EmbeddingGenerator(Generator):
    def __init__(
        self,
        latent_size: int,
        out_shape: Tuple[int, ...],
        num_classes: int,
        name: str = None,
    ):
        Generator.__init__(self, latent_size, out_shape, name)
        self.num_classes = num_classes


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

        Generator.__init__(self, latent_size, out_shape)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout

        x = Input((latent_size,))
        y = self.create_mlp_interim(
            x,
            out_shape,
            latent_size,
            num_layers,
            layer_multiplier,
            bn_momentum,
            leaky_relu_alpha,
            dropout,
        )
        y = Dense(numpy.prod(out_shape), activation="tanh")(y)
        y = Reshape(out_shape)(y)

        Model.__init__(self, inputs=x, outputs=y, name=name or self.__class__.__name__)

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


class EmbeddingMlpGen(EmbeddingGenerator):
    CONDITIONAL = True
    INFINITE = False

    def __init__(
        self,
        latent_size: int,
        num_classes: int,
        out_shape: Tuple[int, ...],
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        leaky_relu_alpha: float,
        dropout: float,
        name: str = None,
    ):

        EmbeddingGenerator.__init__(self, latent_size, out_shape, num_classes, name)

        self.num_layers = num_layers
        self.layer_multiplier = layer_multiplier
        self.bn_momentum = bn_momentum
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = dropout

        latent = Input((latent_size,))
        label = Input((1,), dtype="int32")

        label_embedding = Flatten()(Embedding(num_classes, latent_size)(label))

        x = multiply([latent, label_embedding])
        y = self.create_mlp_interim(
            x,
            out_shape,
            latent_size,
            num_layers,
            layer_multiplier,
            bn_momentum,
            leaky_relu_alpha,
            dropout,
        )
        y = Dense(numpy.prod(out_shape), activation="tanh")(x)
        y = Reshape(out_shape)(y)

        Model.__init__(
            self,
            inputs=[latent, label],
            outputs=y,
            name=name or self.__class__.__name__,
        )

    def create_param_dict(self) -> Dict[str, Any]:
        return {
            Param.LATENT_SIZE.value: self.latent_size,
            Param.OUT_SHAPE.value: self.out_shape,
            Param.N_LAYERS.value: self.num_layers,
            Param.LAYER_MULTIPLIER.value: self.layer_multiplier,
            Param.BATCH_NORM_MOMENTUM.value: self.bn_momentum,
            Param.LEAKY_RELU_ALPHA.value: self.leaky_relu_alpha,
            Param.DROPOUT.value: self.dropout,
            Param.NUM_CLASSES.value: self.num_classes,
        }
