import os
from pathlib import Path

from .data.maker.wisdm import WisdmMaker

WISDM_ROOT_PATH = Path(os.environ["DATASET_ROOT_WISDM"])
INTERIM_ROOT = Path(os.environ["INTERIM_ROOT"])


def main() -> None:
    wisdm_maker = WisdmMaker(WISDM_ROOT_PATH, INTERIM_ROOT)
    wisdm_maker.convert()


if __name__ == "__main__":
    main()
