import os
from pathlib import Path

from .data.datamanager import DataManager
from .data.dataset import Wisdm

WISDM_PATH = Path(os.environ["DATASET_WISDM"])
INTERIM_ROOT = Path(os.environ["INTERIM_ROOT"])


def main() -> None:
    #  wisdm_maker = WisdmMaker(WISDM_ROOT_PATH, INTERIM_ROOT)
    #  wisdm_maker.convert()

    dataset = Wisdm(WISDM_PATH)
    data_manager = DataManager(dataset)

    print(data_manager.read_schema())


if __name__ == "__main__":
    main()
