from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, ClassVar, Dict, Generic, Iterable, Tuple, TypeVar

import mlflow
import mlflow.keras
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
from .discriminator import Discriminator, SimpleMlpDisc
from .generator import Generator, SimpleMlpGen

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


class Metric(Enum):
    GENERATOR_LOSS = "generator_loss"
    DISCRIMINATOR_LOSS = "discriminator_loss"
    DISCRIMINATOR_ACCURACY = "discriminator_accuracy"


class Tag(Enum):
    MLFLOW_NOTE = "mlflow.note.content"
    INFINITE = "infinite"
    CONDITIONAL = "conditional"


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
        smoothing_type: SmoothingType = SmoothingType.NONE,
    ) -> None:
        if generator.CONDITIONAL != discriminator.CONDITIONAL:
            raise ValueError(
                "Generator and Discriminator have to have same conditionality"
            )

        self.smoothing_type = smoothing_type
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.combined = self.combine(optimizer)

        mlflow.start_run(run_name=self.run_name)

        params = self.create_param_dict()
        params.update(self.generator.create_param_dict())
        params.update(self.discriminator.create_param_dict())
        params.update({Param.NUM_CHECKPOINTS.value: self.N_CHECKPOINTS})
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
            new_element = numpy.random.normal(size=(1, self.generator.input_shape[1]))
            new_element /= numpy.linalg.norm(new_element)
            if latents is not None:
                latents = numpy.concatenate((latents, new_element))  # type: ignore
            else:
                latents = new_element
        return latents

    def create_param_dict(self) -> Dict[str, Any]:
        return {
            Param.OPTIMIZER_TYPE.value: self.optimizer.__class__.__name__,
            Param.OPTIMIZER_PARAMS.value: self.optimizer.get_config(),
            Param.SMOOTHING_TYPE.value: self.smoothing_type.name,
        }

    def train(self, data: NumpySequences, num_epochs: int) -> None:
        # Adversarial ground truths
        mlflow.log_params(
            {
                Param.WINDOW.value: data.shape[-2],
                Param.NUM_EPOCH.value: num_epochs,
                Param.BATCH_SIZE.value: data.batch_size,
            }
        )
        ground_real, ground_fake = self.create_ground_values(data.batch_size)

        for epoch in range(num_epochs):
            for batch, (real_sequences, real_classes) in enumerate(data):
                d_loss, d_accuracy, g_loss = self._batch_step(
                    real_sequences, real_classes, ground_real, ground_fake
                )
                print(
                    f"E:{epoch+1:2} | B:{batch+1:<4}/{len(data):<4} [D loss: {d_loss:2.3f}, acc: %{d_accuracy:2}] [G loss: {g_loss:2.3f}]"
                )
                mlflow.log_metrics(
                    {
                        Metric.DISCRIMINATOR_LOSS.value: d_loss,
                        Metric.DISCRIMINATOR_ACCURACY.value: d_accuracy,
                        Metric.GENERATOR_LOSS.value: g_loss,
                    }
                )
                training_step += 1

            samples = self.generate(1)
            dists = measure(samples, data)
            mlflow.log_metrics(
                {
                    Metric.DISTANCE_MIN_EUCLIDEAN.value: dists[0],
                    Metric.DISTANCE_MANHATTAN.value: dists[1],
                },
                epoch,
            )
            # dist = dynamic_time_warp(samples, data)
            # mlflow.log_metric(Metric.DISTANCE_TIME_WARP.value, dist)

            interval = num_epochs // Gan.N_CHECKPOINTS
            if (epoch + 1) % interval == 0:
                mlflow.keras.log_model(
                    self.combined, f"models/gan_{(epoch+1)//interval}"
                )
                mlflow.keras.log_model(
                    self.generator, f"models/generator_{(epoch+1)//interval}"
                )
                mlflow.keras.log_model(
                    self.discriminator, f"models/discriminator_{(epoch+1)//interval}"
                )

        mlflow.end_run()

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

    def generate(self, num_samples: int) -> numpy.ndarray:
        latents = self.generate_latents(num_samples)
        return self.generator.predict(latents)


class SimpleGan(Gan[G, D]):

    DESCRIPTION = "Plain, vanilla GAN without any class information"

    def __init__(self, generator: G, discriminator: D, optimizer: Optimizer,) -> None:
        Gan.__init__(self, generator, discriminator, optimizer)

    def create_param_dict(self) -> Dict[str, Any]:
        return {
            Param.OPTIMIZER_TYPE.value: self.optimizer.__class__.__name__,
            Param.OPTIMIZER_PARAMS.value: self.optimizer.get_config(),
            Param.SMOOTHING_TYPE.value: self.smoothing_type.name,
        }

    def combine(self, optimizer: Optimizer) -> Model:
        latent = Input(shape=(self.generator.input_shape[1],))
        generated_sequence = self.generator(latent)
        self.discriminator.trainable = False
        discrimination = self.discriminator(generated_sequence)
        model = Model(latent, discrimination)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        return model

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
