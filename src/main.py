from pathlib import Path

import toml

from .data.datamanager import DataManager
from .data.dataset import Wisdm, MotionSense

datasets = toml.load("config.toml")["dataset"]
WISDM_PATH = Path(datasets["wisdm"])
MOTION_SENSE_PATH = Path(datasets["motion-sense"])


def main() -> None:
    #  wisdm_maker = WisdmMaker(WISDM_ROOT_PATH, INTERIM_ROOT)
    #  wisdm_maker.convert()

    dataset = MotionSense(MOTION_SENSE_PATH)
    data_manager = DataManager(dataset)

    print(data_manager.read_schema())


if __name__ == "__main__":
    main()
