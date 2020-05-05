from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, Tuple

import numpy
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.utils import Sequence as KerasSequence

from ..data.window import NumpySequences


class Gan(ABC):

    GENERATOR_NAME: ClassVar[str] = "Generator"
    DISCRIMINATOR_NAME: ClassVar[str] = "Discriminator"
    COMBINED_GAN_NAME: ClassVar[str] = "GAN"

    def __init__(self, sequence_shape: Tuple[int, ...], latent_size: int) -> None:
        self.shape = sequence_shape
        self.latent_size = latent_size
        self.generator = self.build_generator(latent_size, sequence_shape)
        self.generator.name = self.GENERATOR_NAME
        self.discriminator = self.build_discriminator(sequence_shape)
        self.discriminator.name = self.DISCRIMINATOR_NAME
        self.combined = self.combine()
        self.combined.name = self.COMBINED_GAN_NAME

    @staticmethod
    def generate_latents(size: int, n_samples: int) -> numpy.ndarray:
        return numpy.random.normal(0, 1, (n_samples, size))

    @classmethod
    @abstractmethod
    def build_generator(cls, latent_size: int, out_shape: Tuple[int, ...]) -> Model:
        pass

    @classmethod
    @abstractmethod
    def build_discriminator(cls, input_shape: Tuple[int, ...]) -> Model:
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


class SimpleGan(Gan):
    def __init__(self, sequence_shape: Tuple[int, ...], latent_size: int) -> None:
        Gan.__init__(self, sequence_shape, latent_size)

    @classmethod
    def build_generator(cls, latent_size: int, out_shape: Tuple[int, ...]) -> Model:
        model = Sequential(
            [
                Dense(256, input_dim=latent_size),
                BatchNormalization(momentum=0.9),
                LeakyReLU(alpha=0.2),
                Dense(1024),
                BatchNormalization(momentum=0.9),
                LeakyReLU(alpha=0.2),
                Dropout(0.5),
                Dense(2048),
                BatchNormalization(momentum=0.9),
                LeakyReLU(alpha=0.2),
                Dropout(0.5),
                Dense(numpy.prod(out_shape), activation="tanh"),
                Reshape(out_shape),
            ]
        )

        return model

    @classmethod
    def build_discriminator(cls, input_shape: Tuple[int, ...]) -> Model:

        optimizer = SGD(0.001)

        model = Sequential(
            [
                Flatten(input_shape=input_shape),
                Dense(1024),
                LeakyReLU(alpha=0.2),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model

    def combine(self) -> Model:
        optimizer = Adam(0.0002, 0.5)

        latent = Input(shape=(self.latent_size,))
        generated_sequence = self.generator(latent)

        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        discrimination = self.discriminator(generated_sequence)

        model = Model(latent, discrimination)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        return model

    # def train(self, data: KerasSequence, num_epochs: int) -> History:
    #     return self.combined.fit(
    #         x=data, epochs=num_epochs, callbacks=[TerminateOnNaN()]
    #     )

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
                    real_sequences, ground_real
                )
                d_loss_fake = self.discriminator.train_on_batch(
                    generated_sequences, ground_fake
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
