from abc import ABC, abstractmethod
from typing import Tuple, Iterable

import numpy
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Dense, Model, Sequential
from keras.optimizers import Adam


class Gan(ABC):
    def __init__(self, sequence_shape: Tuple[int, ...], latent_size: int) -> None:
        self.shape = sequence_shape
        self.latent_size = latent_size
        self.generator = self.build_generator(latent_size, sequence_shape)
        self.discriminator = self.build_discriminator(sequence_shape)
        self.combined = self.combine(self.generator, self.discriminator)

    @staticmethod
    def generate_latents(size: int, n_samples: int) -> numpy.ndarray:
        return np.random.normal(0, 1, (n_samples, size))

    @classmethod
    @abstractmethod
    def build_generator(cls, latent_size: int, out_shape: Tuple[int, ...]) -> Model:
        pass

    @classmethod
    @abstractmethod
    def build_discriminator(cls, input_shape: Tuple[int, ...]) -> Model:
        pass

    @classmethod
    @abstractmethod
    def combine(cls, generator: Model, discriminator: Model) -> Model:
        pass

    @abstractmethod
    def train(self, data: Iterable[numpy.ndarray], num_epochs: int) -> None:
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
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(512),
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(1024),
                LeakyReLU(alpha=0.2),
                BatchNormalization(momentum=0.8),
                Dense(numpy.prod(out_shape), activation="tanh"),
                Reshape(out_shape),
            ]
        )

        # model.summary()

        latent = Input(shape=latent_size)

        return Model(latent, model(latent))

    @classmethod
    def build_discriminator(cls, input_shape: Tuple[int, ...]) -> Model:

        optimizer = Adam(0.0002, 0.5)

        model = Sequential(
            [
                Flatten(input_shape=input_shape),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dense(256),
                LeakyReLU(alpha=0.2),
                Dense(1, activation="sigmoid"),
            ]
        )

        # model.summary()

        sequence = Input(shape=input_shape)

        model = Model(sequence, model(sequence))

        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model

    @classmethod
    def combine(cls, generator: Model, discriminator: Model) -> Model:
        optimizer = Adam(0.0002, 0.5)

        latent = Input(shape=generator.input_shape)
        generated_sequence = generator(latent)

        discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        discrimination = discriminator(generated_sequence)

        model = Model(latent, discrimination)

        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model

    def train(self, data: Iterable[numpy.ndarray], latent_size:int num_epochs: int, batch_size=1) -> None:

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(num_epochs):
            for datum in data:

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent = Gan.generate_latents(latent_size, batch_size)

            generated = self.generator.predict(latent)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(generated, real)
            d_loss_fake = self.discriminator.train_on_batch(generated, fake)
            discriminator_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = Gan.generate_latents(latent_size, batch_size)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(latent, real)

            # Plot the progress
            print(f"{epoch}: [D loss: {d_loss[0]}, acc: {d_loss[1] * 100}] [G loss: {g_loss}]"))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)



