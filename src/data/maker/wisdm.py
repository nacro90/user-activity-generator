import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import arff  # type: ignore

from ..integrity import recursive_sha256, str_sha256
from .maker import Maker

ARFF = Dict[str, Union[str, Any]]


class WisdmMaker(Maker):

    ARFF_FILE = "WISDM_ar_v1.1_transformed_fixed.arff"

    def __init__(self, raw_path: Path, out_path: Path) -> None:
        Maker.__init__(self, raw_path, out_path)
        self.arff_path = raw_path / self.ARFF_FILE

    @property
    def out_file(self) -> Path:
        return (self.out_path / self.hash).with_suffix(".json")

    @property
    def hash(self) -> str:
        return recursive_sha256(self.arff_path)

    def pre(self) -> bool:
        """
        Returns:
            bool: Whether processed dataset exists or not. `True` if dataset has been
                processed before
        """
        if self.out_file.exists():
            return True

        if not self.out_path.exists():
            self.out_path.mkdir(mode=0o755, parents=True)

        return False

    def make(self) -> None:
        content: ARFF = arff.load(self.arff_path.open("r"))
        self.out = {}
        self.out["rawHash"] = self.hash
        self.out["createdAt"] = self.createTimestamp()
        self.out["data"] = content["data"]

    def commit(self) -> None:
        digest = str_sha256(
            json.dumps(self.out, indent=int(os.environ["JSON_INDENTATION"]))
        )
        self.out["interimHash"] = digest

    def post(self) -> None:
        json.dump(
            self.out,
            self.out_file.open("w"),
            indent=int(os.environ["JSON_INDENTATION"]),
        )

    def __len__(self) -> int:
        return len(arff.load(self.arff_path.open("r"))["data"])
