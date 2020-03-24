import hashlib
from pathlib import Path
from typing import Any


def recursive_sha256(path: Path, s: Any = None) -> str:
    if not path.exists():
        raise ValueError("Path does not exist")

    s = hashlib.sha256() if not s else s

    if path.is_dir():
        for item in path.iterdir():
            recursive_sha256(item, s)
    else:
        s.update(path.read_bytes())

    return str(s.hexdigest())
