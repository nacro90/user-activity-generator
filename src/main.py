from pathlib import Path

import toml

from .data.datamanager import DataManager
from .data.dataset import Activity, MotionSense, Wisdm

datasets = toml.load("config.toml")["dataset"]
WISDM_PATH = Path(datasets["wisdm"])
MOTION_SENSE_PATH = Path(datasets["motion-sense"])


def main() -> None:
    import mlflow
    import mlflow.keras
    import numpy
    import keras
    from keras import Sequential
    from keras.layers import Dense

    # numpy.random.seed(1)

    # a = numpy.random.random((26,26))
    # b = numpy.random.random((3,))

    # numpy.set_printoptions(formatter={"float": f"{: 0.3f}"})

    # with open("/tmp/osman.txt", "w") as file:
    #     file.write(str(a))

    # mlflow.log_artifact("/tmp/osman.txt")

    # seq = Sequential([Dense(512, input_shape=(100,)), Dense(512)])
    # seq.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(0.002))

    # seq2 = Sequential([Dense(12, input_shape=(10,)), Dense(12)])
    # seq2.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(0.002))

    # stringlist = []
    # stringlist.append(f"# {seq.name}")
    # seq.summary(print_fn=lambda x: stringlist.append(f"    {x}"))
    # stringlist.append(f"## Generator")
    # seq2.summary(print_fn=lambda x: stringlist.append(f"    {x}"))
    # stringlist.append(f"## Discriminator")
    # seq2.summary(print_fn=lambda x: stringlist.append(f"    {x}"))
    # short_model_summary = "\n".join(stringlist)

    # mlflow.set_tag("mlflow.note.content", short_model_summary)

    # mlflow.keras.log_model(seq, "models/gan", keras_module="keras")
    # mlflow.keras.log_model(seq, "models/generator", keras_module="keras")
    # mlflow.keras.log_model(seq, "models/discriminator", keras_module="keras")
    # mlflow.keras.log_model(seq, "models/gan", keras_module="keras")
    # mlflow.keras.log_model(seq, "models/generator", keras_module="keras")
    # mlflow.keras.log_model(seq, "models/discriminator", keras_module="keras")

    # Without Experiment

    # with mlflow.start_run(run_name="run w/o experiment") as run:
    #     mlflow.log_param("my", "param")
    #     mlflow.log_metric("score", 1)
    #     mlflow.log_metric("score", 2)
    #     mlflow.log_metric("score", 3)

    # with mlflow.start_run(run_name="run w/o experiment with tag") as run:
    #     mlflow.set_tag("what", "w/o exp tag")
    #     mlflow.log_param("my", "param")
    #     mlflow.log_metric("score", 15)
    #     mlflow.log_metric("score", 3)
    #     mlflow.log_metric("score", 4)

    # with mlflow.start_run(run_name="run w/o experiment") as run:
    #     mlflow.log_param("semsi", "param")
    #     mlflow.log_metric("score", 6)
    #     mlflow.log_metric("score", 2)
    #     mlflow.log_metric("score", -8)

    #     with mlflow.start_run(run_name="nested run w/o experiment with tag", nested=True) as run:
    #         mlflow.set_tag("what", "w/o exp tag nested")
    #         mlflow.log_param("my", "param")
    #         mlflow.log_metric("osman", -16)
    #         mlflow.log_metric("osman", 3)
    #         mlflow.log_metric("osman", 4)

    #     with mlflow.start_run(run_name="nestedededede run w/o experiment with tag", nested=True) as run:
    #         mlflow.set_tag("what", "w/o exp tag nested")
    #         mlflow.log_param("my", "param")
    #         mlflow.log_metric("score", -16)
    #         mlflow.log_metric("score", 3)
    #         mlflow.log_metric("score", 4)

    # mlflow.set_experiment("EXPERIEMENTEEE")

    # with mlflow.start_run(run_name="run w experiment") as run:
    #     mlflow.log_param("my", "param")
    #     mlflow.log_metric("score", 1)
    #     mlflow.log_metric("score", 2)
    #     mlflow.log_metric("score", 3)

    # with mlflow.start_run(run_name="run w experiment with tag") as run:
    #     mlflow.set_tag("what", "w exp tag")
    #     mlflow.log_param("my", "param")
    #     mlflow.log_metric("score", 15)
    #     mlflow.log_metric("score", 3)
    #     mlflow.log_metric("score", 4)

    # with mlflow.start_run(run_name="run w experiment") as run:
    #     mlflow.log_param("my", "param")
    #     mlflow.log_metric("score", 6)
    #     mlflow.log_metric("score", 2)
    #     mlflow.log_metric("score", -8)

    #     with mlflow.start_run(run_name="nested run w experiment with tag", nested=True) as run:
    #         mlflow.set_tag("what", "w exp tag nested")
    #         mlflow.log_param("my", "param")
    #         mlflow.log_metric("score", -16)
    #         mlflow.log_metric("score", 3)
    #         mlflow.log_metric("score", 4)


if __name__ == "__main__":
    main()
