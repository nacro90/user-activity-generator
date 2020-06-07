import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (Any, ClassVar, Dict, Generic, Iterable, Optional, Sequence,
                    Tuple, TypeVar, Union)

import mlflow
import mlflow.keras
import numpy
from keras import backend
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Embedding, Flatten, Input, Lambda, Reshape,
                          ZeroPadding2D, concatenate, multiply)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, Optimizer

from ..data.window import NumpySequences
from ..util.measurement import (create_confusion_matrix,
                                create_epoch_measurements, dynamic_time_warp,
                                measure, min_euclidean)
from .discriminator import (Discriminator, EmbeddingDiscriminator,
                            EmbeddingMlpDisc, LabelingDiscriminator,
                            SimpleMlpDisc)
from .generator import (EmbeddingGenerator, EmbeddingMlpGen, Generator,
                        SimpleMlpGen)

G = TypeVar("G", bound=Generator)
D = TypeVar("D", bound=Discriminator)


class Param(Enum):
    OPTIMIZER_TYPE = "gan_opt_type"
    OPTIMIZER_PARAMS = "gan_opt_params"
    SMOOTHING_TYPE = "smoothing"
    NUM_CHECKPOINTS = "num_checkpoints"
    BATCH_SIZE = "batch_size"
    NUM_EPOCH = "num_epoch"
    WINDOW = "window"
    N_FEATURES = "num_features"


class Metric(Enum):
    GENERATOR_LOSS = "generator_loss"
    DISCRIMINATOR_LOSS = "discriminator_loss"
    DISCRIMINATOR_ACCURACY = "discriminator_accuracy"
    DISTANCE_MIN_EUCLIDEAN = "distance_min_euclidean"
    DISTANCE_MANHATTAN = "distance_manhattan"


class Tag(Enum):
    MLFLOW_NOTE = "mlflow.note.content"
    INFINITE = "infinite"
    CONDITIONAL = "conditional"
    FAILURE = "failure"


class SmoothingType(Enum):
    NONE = auto()
    NOISE = auto()
    PULL_DOWN = auto()


class Gan(ABC, Generic[G, D]):

    N_CHECKPOINTS: ClassVar[int] = 3
    DESCRIPTION: ClassVar[str] = NotImplemented

    def __init__(
        self,
        generator: G,
        discriminator: D,
        optimizer: Optimizer,
        smoothing_type: SmoothingType = SmoothingType.PULL_DOWN,
        max_n_batch: Optional[int] = None,
        num_classes: int = 1,
    ) -> None:
        if generator.CONDITIONAL != discriminator.CONDITIONAL:
            raise ValueError(
                "Generator and Discriminator have to have same conditionality"
            )
        if generator.CONDITIONAL and num_classes == 1:
            raise ValueError("Conditional GANS's must have more than one class")

        self.num_classes = num_classes
        self.smoothing_type = smoothing_type
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.combined = self.combine(optimizer)
        self.max_n_batch = max_n_batch

        mlflow.start_run(run_name=self.run_name)

        params = self.create_param_dict()
        params.update(self.generator.create_param_dict())
        params.update(self.discriminator.create_param_dict())
        mlflow.log_params(params)

        mlflow.set_tag(Tag.MLFLOW_NOTE.value, self.create_note_content())
        mlflow.set_tag(Tag.INFINITE.value, self.generator.INFINITE)
        mlflow.set_tag(Tag.CONDITIONAL.value, self.generator.CONDITIONAL)

    @staticmethod
    def noise_clasess(classes: numpy.ndarray, range: float = 0.2) -> numpy.ndarray:
        rands = numpy.random.normal(size=classes.shape)
        rands /= numpy.linalg.norm(rands)
        noised = classes + rands * range
        return noised.clip(min=0)

    @abstractmethod
    def combine(self, optimizer: Optimizer) -> Model:
        pass

    @property
    def run_name(self) -> str:
        return f"{self.__class__.__name__}[{self.generator.name},{self.discriminator.name}]"

    @abstractmethod
    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:
        pass

    @abstractmethod
    def _create_param_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, num_samples: int, label: int = None) -> numpy.ndarray:
        pass

    def create_note_content(self) -> str:
        lines = []
        lines.append("# Summary\n")
        self.combined.summary(print_fn=lambda x: lines.append(f"    {x}"))
        lines.append("# Generator\n")
        self.generator.summary(print_fn=lambda x: lines.append(f"    {x}"))
        lines.append("# Discriminator\n")
        self.discriminator.summary(print_fn=lambda x: lines.append(f"    {x}"))
        return "\n".join(lines)

    def generate_latents(self, n_samples: int) -> numpy.ndarray:
        latents = None
        for _ in range(n_samples):
            new_element = numpy.random.normal(size=(1, self.generator.latent_size))
            new_element /= numpy.linalg.norm(new_element)
            if latents is not None:
                latents = numpy.concatenate((latents, new_element))  # type: ignore
            else:
                latents = new_element
        return latents

    def generate_labels(self, n_samples: int, onehot: bool) -> numpy.ndarray:
        if onehot:
            return numpy.eye(self.num_classes)[
                numpy.random.choice(self.num_classes, n_samples)
            ]
        return numpy.random.randint(0, self.num_classes, (n_samples, 1))

    def create_param_dict(self) -> Dict[str, Any]:
        params = self._create_param_dict()
        params.update(
            {
                Param.OPTIMIZER_TYPE.value: self.optimizer.__class__.__name__,
                Param.OPTIMIZER_PARAMS.value: self.optimizer.get_config(),
                Param.SMOOTHING_TYPE.value: self.smoothing_type.name,
                Param.NUM_CHECKPOINTS.value: self.N_CHECKPOINTS,
            }
        )
        return params

    def train(self, data: NumpySequences, num_epochs: int) -> None:
        mlflow.log_params(
            {
                Param.WINDOW.value: data.shape[-2],
                Param.NUM_EPOCH.value: num_epochs,
                Param.BATCH_SIZE.value: data.batch_size,
            }
        )
        # Adversarial ground truths
        ground_real, ground_fake = self.create_ground_values(data.batch_size)

        training_step = 0
        for epoch in range(num_epochs):
            for batch, (real_sequences, real_classes) in enumerate(data):
                d_loss, d_accuracy, g_loss = self._batch_step(
                    real_sequences, real_classes, ground_real, ground_fake
                )
                if numpy.isnan(d_loss) or numpy.isnan(g_loss):
                    print(
                        f"NaN's detected in the result: d_loss:{d_loss}, d_accuracy:{d_accuracy}, g_loss:{g_loss}"
                    )
                    mlflow.set_tag(Tag.FAILURE.value, "NaN")
                    return
                print(
                    f"E:{epoch+1:2} | B:{batch+1:>4}/{len(data):<4} [D loss: {d_loss:2.3f}, acc: %{d_accuracy:2}] [G loss: {g_loss:2.3f}]"
                )
                mlflow.log_metrics(
                    {
                        Metric.DISCRIMINATOR_LOSS.value: d_loss,
                        Metric.DISCRIMINATOR_ACCURACY.value: d_accuracy,
                        Metric.GENERATOR_LOSS.value: g_loss,
                    },
                    training_step,
                )

                if self.max_n_batch:
                    if batch == self.max_n_batch:
                        break
                    data.shuffle_indexes()

                training_step += 1

            self.log_epoch(data, epoch)

            interval = num_epochs // Gan.N_CHECKPOINTS
            if (epoch + 1) % interval == 0:
                self.log_checkpoint(data, (epoch + 1) // interval)

        mlflow.end_run()

    def log_epoch(self, data: NumpySequences, epoch: int) -> None:
        if self.num_classes == 1:
            samples = self.generate(1)
        else:
            samples = numpy.array(
                [self.generate(1, i) for i in range(self.num_classes)]
            )
        dist = create_epoch_measurements(samples, data)
        mlflow.log_metric(Metric.DISTANCE_MIN_EUCLIDEAN.value, dist, epoch)

    def log_checkpoint(self, data: NumpySequences, checkpoint_num: int) -> None:
        mlflow.keras.log_model(self.combined, f"models/gan-{checkpoint_num}")
        mlflow.keras.log_model(self.generator, f"models/generator-{checkpoint_num}")
        mlflow.keras.log_model(
            self.discriminator, f"models/discriminator-{checkpoint_num}"
        )

        if self.num_classes > 1:
            filename = f"confusion-{checkpoint_num}.txt"
            with open(filename, "w") as file:
                samples = numpy.array(
                    [self.generate(1, i) for i in range(self.num_classes)]
                )
                confusion_matrix = create_confusion_matrix(samples, data)
                numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})
                file.write(str(confusion_matrix))
                numpy.set_printoptions()
            mlflow.log_artifact(filename)

    def create_ground_values(
        self, n_samples: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # TODO multiclass
        if self.smoothing_type == SmoothingType.PULL_DOWN:
            return numpy.ones((n_samples, 1)) - 0.1, numpy.zeros((n_samples, 1))
        if self.smoothing_type == SmoothingType.NOISE:
            return (
                Gan.noise_clasess(numpy.ones((n_samples, 1))),
                Gan.noise_clasess(numpy.zeros((n_samples, 1))),
            )
        return numpy.ones((n_samples, 1)), numpy.zeros((n_samples, 1))


class SimpleGan(Gan[G, D]):

    DESCRIPTION = "Plain, vanilla GAN without any class information"

    def __init__(
        self,
        generator: G,
        discriminator: D,
        optimizer: Optimizer,
        smoothing_type: SmoothingType = None,
    ) -> None:
        Gan.__init__(self, generator, discriminator, optimizer)
        if smoothing_type:
            self.smoothing_type = smoothing_type

    def _create_param_dict(self) -> Dict[str, Any]:
        return {}

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input(shape=(self.generator.latent_size,))
        generated_sequence = self.generator(latent)
        self.discriminator.trainable = False
        discrimination = self.discriminator(generated_sequence)
        model = Model(latent, discrimination)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        return model

    def generate(self, num_samples: int, _: int = None) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        return self.generator.predict(latents)

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        # ---------------------
        #  Train Discriminator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])

        generated_sequences = self.generator.predict(latents)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(real_sequences, ground_real)
        d_loss_fake = self.discriminator.train_on_batch(
            generated_sequences, ground_fake
        )
        d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
        d_accuracy = int(round(d_accuracy * 100))

        # ---------------------
        #  Train Generator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])
        g_loss = self.combined.train_on_batch(latents, ground_real)

        return d_loss, d_accuracy, g_loss


G_E = TypeVar("G_E", bound=EmbeddingGenerator)
D_E = TypeVar("D_E", bound=EmbeddingDiscriminator)


class CGan(Gan[G_E, D_E]):

    DESCRIPTION = "Conditional GAN"

    def __init__(
        self,
        num_classes: int,
        generator: G_E,
        discriminator: D_E,
        optimizer: Optimizer,
        smoothing_type: SmoothingType = None,
    ) -> None:
        if smoothing_type:
            self.smoothing_type = smoothing_type

        Gan.__init__(self, generator, discriminator, optimizer, num_classes=num_classes)

    def _create_param_dict(self) -> Dict[str, Any]:
        return {}

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input((self.generator.latent_size,))
        label = Input((1,))

        generated_sequence = self.generator([latent, label])

        self.discriminator.trainable = False

        discrimination = self.discriminator([generated_sequence, label])

        model = Model([latent, label], discrimination)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        return model

    def generate(self, num_samples: int, label: int = None) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        labels = (
            numpy.array([label] * num_samples)
            if label
            else numpy.random.randint(0, self.num_classes, num_samples).reshape(-1, 1)
        )

        return self.generator.predict([latents, labels])

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        labels = real_classes.argmax(axis=-1)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])

        generated_sequence = self.generator.predict([latents, labels])

        d_loss_real = self.discriminator.train_on_batch(
            [real_sequences, labels], ground_real
        )
        d_loss_fake = self.discriminator.train_on_batch(
            [generated_sequence, labels], ground_fake
        )
        d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
        d_accuracy = int(round(d_accuracy * 100))

        # ---------------------
        #  Train Generator
        # ---------------------

        sampled_labels = numpy.random.randint(
            0, self.num_classes, (real_sequences.shape[0], 1)
        )

        g_loss = self.combined.train_on_batch([latents, sampled_labels], ground_real)

        return d_loss, d_accuracy, g_loss


D_L = TypeVar("D_L", bound=LabelingDiscriminator)


class AcGan(CGan[G_E, D_L]):

    DESCRIPTION = "Auxiliary Classifier GAN"

    def __init__(
        self,
        num_classes: int,
        generator: G_E,
        discriminator: D_L,
        optimizer: Optimizer,
        smoothing_type: SmoothingType = None,
    ) -> None:
        CGan.__init__(
            self,
            num_classes,
            generator,
            discriminator,
            optimizer,
            smoothing_type,
        )

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input((self.generator.latent_size,))
        target_label = Input((1,))

        generated_sequence = self.generator([latent, target_label])

        self.discriminator.trainable = False

        discrimination, predicted_label = self.discriminator(generated_sequence)

        model = Model([latent, target_label], [discrimination, predicted_label])
        model.compile(
            loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
            optimizer=optimizer,
        )

        return model

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        labels = real_classes.argmax(axis=-1).reshape(-1, 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])

        sampled_labels = self.generate_labels(real_sequences.shape[0], False)

        generated_sequence = self.generator.predict([latents, labels])

        d_loss_real = self.discriminator.train_on_batch(
            real_sequences, [ground_real, labels]
        )
        d_loss_fake = self.discriminator.train_on_batch(
            generated_sequence, [ground_fake, sampled_labels]
        )
        d_loss, *_, disc_accuracy, cls_accuracy = 0.5 * numpy.add(
            d_loss_real, d_loss_fake
        )
        d_accuracy = (disc_accuracy + cls_accuracy) / 2
        d_accuracy = int(round(d_accuracy * 100))

        # ---------------------
        #  Train Generator
        # ---------------------

        g_loss = self.combined.train_on_batch(
            [latents, sampled_labels], [ground_real, sampled_labels]
        )

        return d_loss, d_accuracy, g_loss[0]


class WGan(Gan[G, D]):  # TODO Use with RMSprop(lr=0.00005)

    DESCRIPTION: ClassVar[str] = "Wasserstein GAN"
    CLIP_VALUE: ClassVar[float] = 0.01

    def __init__(
        self,
        generator: G,
        discriminator: D,
        optimizer: Optimizer,
        max_n_batch: int,
        smoothing_type: SmoothingType = None,
    ) -> None:

        Gan.__init__(
            self, generator, discriminator, optimizer,
        )

        if smoothing_type:
            self.smoothing_type = smoothing_type

        self.discriminator.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )

    def _create_param_dict(self) -> Dict[str, Any]:
        return {}

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input((self.generator.latent_size,))

        generated_sequence = self.generator(latent)

        self.discriminator.trainable = False

        discrimination = self.discriminator(generated_sequence)

        model = Model(latent, discrimination)
        model.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    def wasserstein_loss(self, y_true: Any, y_pred: Any) -> Any:
        return backend.mean(y_true * y_pred)

    def generate(self, num_samples: int, label: int = None) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        labels = (
            numpy.array([label] * num_samples)
            if label
            else numpy.random.randint(0, self.num_classes, num_samples).reshape(-1, 1)
        )

        return self.generator.predict(latents)

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        labels = real_classes.argmax(axis=-1)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])

        generated_sequence = self.generator.predict(latents)

        d_loss_real = self.discriminator.train_on_batch(real_sequences, ground_real)
        d_loss_fake = self.discriminator.train_on_batch(generated_sequence, ground_fake)
        d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
        d_accuracy = int(round(d_accuracy * 100))

        # Clip critic weights
        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [
                numpy.clip(w, -WGan.CLIP_VALUE, WGan.CLIP_VALUE) for w in weights
            ]
            l.set_weights(weights)

        # ---------------------
        #  Train Generator
        # ---------------------

        g_loss = self.combined.train_on_batch(latents, ground_real)

        return 1 - d_loss, d_accuracy, 1 - g_loss[0]


class AdversarialAutoencoder(Gan[G, D]):

    DESCRIPTION = "Adversarial Autoencoder. Discriminates between encoded data and random latent numbers"

    def __init__(
        self,
        generator: G,  # Generator is a decoder
        discriminator: D,
        optimizer: Optimizer,
        num_encoding_layer: int,
        smoothing_type: SmoothingType = None,
    ) -> None:
        if smoothing_type:
            self.smoothing_type = smoothing_type

        self.num_encoding_layer = num_encoding_layer
        self.encoder = self.build_encoder(generator)

        Gan.__init__(self, generator, discriminator, optimizer)

    def build_encoder(self, generator: G) -> Model:
        def sampling(args: Any) -> Any:
            z_mean, z_log_var = args
            batch = backend.shape(z_mean)[0]
            dim = backend.int_shape(z_mean)[1]
            epsilon = backend.random_normal(shape=(batch, dim))
            return z_mean + backend.exp(0.5 * z_log_var) * epsilon

        x = Input(generator.out_shape)
        y = Flatten()(x)
        for _ in range(self.num_encoding_layer):
            y = Dense(numpy.prod(generator.out_shape))(y)
            y = LeakyReLU(alpha=generator.leaky_relu_alpha)(y)
        mu = Dense(generator.latent_size)(y)
        log_var = Dense(generator.latent_size)(y)
        latent_repr = Lambda(
            sampling,
            output_shape=(generator.latent_size,),
            name="latent_representation",
        )([mu, log_var])

        return Model(x, latent_repr)

    def _create_param_dict(self) -> Dict[str, Any]:
        return {}

    def combine(self, optimizer: Optimizer) -> Model:
        x = Input(self.generator.out_shape)
        encoded_x = self.encoder(x)
        decoded_x = self.generator(encoded_x)
        self.discriminator.trainable = False
        discrimination = self.discriminator(encoded_x)
        model = Model(x, [decoded_x, discrimination])
        model.compile(
            loss=["mse", "binary_crossentropy"],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer,
        )
        return model

    def generate(self, num_samples: int, _: int = None) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        return self.generator.predict(latents)

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # generated_sequences = self.generator.predict(latents)

        latent_fake = self.encoder.predict(real_sequences)
        latent_real = self.generate_latents(real_sequences.shape[0])

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(latent_real, ground_real)
        d_loss_fake = self.discriminator.train_on_batch(latent_fake, ground_fake)
        d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
        d_accuracy = int(round(d_accuracy * 100))

        # ---------------------
        #  Train Generator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])
        g_loss = self.combined.train_on_batch(
            real_sequences, [real_sequences, ground_real]
        )

        return d_loss, d_accuracy, g_loss[0]


class VaeGan(Gan[G, D]):

    DESCRIPTION = "Variational Adversarial Autoencoder. Discriminates between encoded-decoded data and decoded random latent numbers"

    def __init__(
        self,
        generator: G,  # Generator is a decoder
        discriminator: D,
        optimizer: Optimizer,
        num_encoding_layer: int,
        smoothing_type: SmoothingType = None,
    ) -> None:
        if smoothing_type:
            self.smoothing_type = smoothing_type

        self.num_encoding_layer = num_encoding_layer
        self.encoder = self.build_encoder(generator)

        Gan.__init__(self, generator, discriminator, optimizer)

    def build_encoder(self, generator: G) -> Model:
        def sampling(args: Any) -> Any:
            z_mean, z_log_var = args
            batch = backend.shape(z_mean)[0]
            dim = backend.int_shape(z_mean)[1]
            epsilon = backend.random_normal(shape=(batch, dim))
            return z_mean + backend.exp(0.5 * z_log_var) * epsilon

        x = Input(generator.out_shape)
        y = Flatten()(x)
        for _ in range(self.num_encoding_layer):
            y = Dense(numpy.prod(generator.out_shape))(y)
            y = LeakyReLU(alpha=generator.leaky_relu_alpha)(y)
        mu = Dense(generator.latent_size)(y)
        log_var = Dense(generator.latent_size)(y)
        latent_repr = Lambda(
            sampling,
            output_shape=(generator.latent_size,),
            name="latent_representation",
        )([mu, log_var])

        return Model(x, latent_repr)

    def _create_param_dict(self) -> Dict[str, Any]:
        return {}

    def combine(self, optimizer: Optimizer) -> Model:
        x = Input(self.generator.out_shape)
        encoded_x = self.encoder(x)
        decoded_x = self.generator(encoded_x)
        self.discriminator.trainable = False
        discrimination = self.discriminator(decoded_x)
        model = Model(x, [decoded_x, discrimination])
        model.compile(
            loss=["mse", "binary_crossentropy"],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer,
        )
        return model

    def generate(self, num_samples: int, _: int = None) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        return self.generator.predict(latents)

    def _batch_step(
        self,
        real_sequences: numpy.ndarray,
        real_classes: numpy.ndarray,
        ground_real: numpy.ndarray,
        ground_fake: numpy.ndarray,
    ) -> Tuple[float, int, float]:

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # generated_sequences = self.generator.predict(latents)

        latent_fake = self.encoder.predict(real_sequences)
        latent_real = self.generate_latents(real_sequences.shape[0])

        decoded_real = self.generator.predict(latent_fake)
        decoded_latent = self.generator.predict(latent_real)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(decoded_real, ground_real)
        d_loss_fake = self.discriminator.train_on_batch(decoded_latent, ground_fake)
        d_loss, d_accuracy = 0.5 * numpy.add(d_loss_real, d_loss_fake)
        d_accuracy = int(round(d_accuracy * 100))

        # ---------------------
        #  Train Generator
        # ---------------------

        latents = self.generate_latents(real_sequences.shape[0])
        g_loss = self.combined.train_on_batch(
            real_sequences, [real_sequences, ground_real]
        )

        return d_loss, d_accuracy, g_loss[0]
