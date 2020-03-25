import hashlib
from pathlib import Path
from typing import Any


def recursive_sha256(path: Path, hashsum: Any = None) -> str:
    if not path.exists():
        raise ValueError("Path does not exist")

    hashsum = hashlib.sha256() if not hashsum else hashsum

    if path.is_dir():
        for item in path.iterdir():
            recursive_sha256(item, hashsum)
    else:
        hashsum.update(path.read_bytes())

    return str(hashsum.hexdigest())
