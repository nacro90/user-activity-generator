from pathlib import Path
from typing import ClassVar

import toml

config = toml.load("config.toml")["config"]


class Config:
    INTERIM_ROOT: ClassVar[Path] = Path(config["interim-root"])
    HASH_LENGTH: ClassVar[int] = int(config["hash-length"])
