from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Iterable, Tuple, TypeVar

import numpy
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, Optimizer

from ..data.window import NumpySequences
from .discriminator import Discriminator, MlpDisc
from .generator import Generator, MlpGen

G = TypeVar("G", bound=Generator)
D = TypeVar("D", bound=Discriminator)


class Gan(ABC):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        sequence_shape: Tuple[int, ...],
        latent_size: int,
    ) -> None:
        self.shape = sequence_shape
        self.latent_size = latent_size
        self.generator = generator
        self.discriminator = discriminator
        self.combined = self.combine()

    @staticmethod
    def generate_latents(size: int, n_samples: int) -> numpy.ndarray:
        return numpy.random.normal(0, 1, (n_samples, size))

    @staticmethod
    def noise_clasess(classes: numpy.ndarray, range: float = 0.3) -> numpy.ndarray:
        rands = (
            numpy.random.randn(numpy.prod(classes.shape)).reshape(classes.shape)
            * range
            * 2
        )
        noised = classes - range + rands
        return noised.clip(min=0)

    @classmethod
    @abstractmethod
    def build_generator(
        cls,
        latent_size: int,
        out_shape: Tuple[int, ...],
        layers: int,
        layer_multiplier: float,
        momentum: float,
        leaky_relu_alpha: float,
        dropout: float,
    ) -> G:
        pass

    @classmethod
    @abstractmethod
    def build_discriminator(
        self,
        in_shape: Tuple[int, ...],
        num_classes: int,
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        dropout: float,
        leaky_relu_alpha: float,
        optimizer: Optimizer,
    ) -> MlpDisc:
        pass

    @abstractmethod
    def combine(self) -> Model:
        pass

    @abstractmethod
    def train(self, data: NumpySequences, num_epochs: int) -> None:
        pass

    def generate(self, num_samples: int) -> numpy.ndarray:
        latents = self.generate_latents(self.latent_size, num_samples)
        return self.generator.predict(latents)


class SimpleGan(Gan[MlpGen, MlpDisc]):
    def __init__(self, sequence_shape: Tuple[int, ...], latent_size: int) -> None:
        Gan.__init__(self, sequence_shape, latent_size)

    @classmethod
    def build_generator(
        cls,
        latent_size: int,
        out_shape: Tuple[int, ...],
        layers: int,
        layer_multiplier: float,
        momentum: float,
        leaky_relu_alpha: float,
        dropout: float,
    ) -> MlpGen:
        return MlpGen(
            latent_size,
            out_shape,
            layers,
            layer_multiplier,
            momentum,
            dropout,
            leaky_relu_alpha,
        )

    @classmethod
    def build_discriminator(
        self,
        in_shape: Tuple[int, ...],
        num_classes: int,
        num_layers: int,
        layer_multiplier: float,
        bn_momentum: float,
        dropout: float,
        leaky_relu_alpha: float,
        name: str,
    ) -> MlpDisc:

        return MlpDisc(
            in_shape,
            num_classes,
            num_layers,
            layer_multiplier,
            bn_momentum,
            dropout,
            leaky_relu_alpha,
            name,
        )

    def combine(self, optimizer: Optimizer) -> Model:

        latent = Input(shape=(self.latent_size,))
        generated_sequence = self.generator(latent)

        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        discrimination = self.discriminator(generated_sequence)

        model = Model(latent, discrimination)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        return model

    def train(self, data: NumpySequences, num_epochs: int) -> None:

        # Adversarial ground truths
        ground_real = numpy.ones((data.batch_size, 1))
        ground_fake = numpy.zeros((data.batch_size, 1))

        for epoch in range(num_epochs):
            for batch, (real_sequences, real_classes) in enumerate(data):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                latents = self.generate_latents(self.latent_size, data.batch_size)

                generated_sequences = self.generator.predict(latents)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(
                    real_sequences, Gan.noise_clasess(ground_real)
                )
                d_loss_fake = self.discriminator.train_on_batch(
                    generated_sequences, Gan.noise_clasess(ground_fake)
                )
                d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
                d_accuracy = int(round(d_accuracy * 100))

                # ---------------------
                #  Train Generator
                # ---------------------

                latents = self.generate_latents(self.latent_size, data.batch_size)

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(latents, ground_real)

                # Plot the progress
                print(
                    f"{epoch:2}:{batch:4} [D loss: {d_loss:2.3f}, acc: {d_accuracy:3}] [G loss: {g_loss:2.3f}]"
                )

                # # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_images(epoch)
