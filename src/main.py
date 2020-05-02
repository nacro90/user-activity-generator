from pathlib import Path

import toml

from .data.datamanager import DataManager
from .data.dataset import Activity, MotionSense, Wisdm

datasets = toml.load("config.toml")["dataset"]
WISDM_PATH = Path(datasets["wisdm"])
MOTION_SENSE_PATH = Path(datasets["motion-sense"])


def main() -> None:
    #  wisdm_maker = WisdmMaker(WISDM_ROOT_PATH, INTERIM_ROOT)
    #  wisdm_maker.convert()

    dataset = MotionSense(MOTION_SENSE_PATH)
    data_manager = DataManager(dataset)

    return data_manager.stream({Activity.WALKING}, dataset.FREQUENCY * 2)


if __name__ == "__main__":
    main()
