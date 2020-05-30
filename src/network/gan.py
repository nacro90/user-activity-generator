from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Iterable, Tuple, TypeVar

import numpy
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    ZeroPadding2D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, Optimizer

from ..data.window import NumpySequences
from .discriminator import Discriminator, NonConditionalMlpDisc
from .generator import Generator, NonConditionalMlpGen

G = TypeVar("G", bound=Generator)
D = TypeVar("D", bound=Discriminator)


class Gan(ABC, Generic[G, D]):
    def __init__(self, generator: G, discriminator: D, optimizer: Optimizer,) -> None:
        if generator.CONDITIONAL != discriminator.CONDITIONAL:
            raise ValueError(
                "Generator and Discriminator have have same conditionality"
            )
        self.generator = generator
        self.discriminator = discriminator
        self.combined = self.combine(optimizer)

    def generate_latents(self, n_samples: int) -> numpy.ndarray:
        return numpy.random.normal(0, 1, (n_samples, self.generator.input_shape[1]))

    @staticmethod
    def noise_clasess(classes: numpy.ndarray, range: float = 0.3) -> numpy.ndarray:
        rands = (
            numpy.random.randn(numpy.prod(classes.shape)).reshape(classes.shape)
            * range
            * 2
        )
        noised = classes - range + rands
        return noised.clip(min=0)

    @abstractmethod
    def combine(self, optimizer: Optimizer) -> Model:
        pass

    @abstractmethod
    def train(self, data: NumpySequences, num_epochs: int) -> None:
        pass

    def generate(self, num_samples: int) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        return self.generator.predict(latents)


class SimpleGan(Gan[G, D]):
    def __init__(self, generator: G, discriminator: D, optimizer: Optimizer,) -> None:
        Gan.__init__(self, generator, discriminator, optimizer)

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input(shape=(self.generator.input_shape[1],))
        generated_sequence = self.generator(latent)
        self.discriminator.trainable = False
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

                latents = self.generate_latents(data.batch_size)

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

                latents = self.generate_latents(data.batch_size)

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(latents, ground_real)

                # Plot the progress
                print(
                    f"E:{epoch:2} | B:{batch:<4}/{len(data):4} [D loss: {d_loss:2.3f}, acc: %{d_accuracy:2}] [G loss: {g_loss:2.3f}]"
                )

                # # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_images(epoch)

                # # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_images(epoch)
