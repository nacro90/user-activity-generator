import toml
from pathlib import Path

from .data.datamanager import DataManager
from .data.dataset import Wisdm

datasets = toml.load("config.toml")["dataset"]
WISDM_PATH = Path(datasets["wisdm"])

def main() -> None:
    #  wisdm_maker = WisdmMaker(WISDM_ROOT_PATH, INTERIM_ROOT)
    #  wisdm_maker.convert()

    dataset = Wisdm(WISDM_PATH)
    data_manager = DataManager(dataset)

    print(data_manager.read_schema())


if __name__ == "__main__":
    main()
