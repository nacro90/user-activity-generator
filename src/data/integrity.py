"""
File: integrity.py
Author: yourname
Email: yourname@email.com
Github: https://github.com/yourname
Description: Checks the integrity of the data using hashes
"""

import hashlib
from pathlib import Path
from typing import Any


def recursive_sha256(path: Path, hashsum: Any = None) -> str:
    """
    Calculates sha256 hash of the file contents recursively.

    Args:
        path (Path): Parent path of contents
        hashsum (Optional[hashlib._HASH]): Current checksum of files if any

    Returns:
        str: Accumulated digest hex number string

    Raises:
        ValueError: When `path` does not exist in the system
    """
    if not path.exists():
        raise ValueError("Path does not exist")

    hashsum = hashlib.sha256() if not hashsum else hashsum

    if path.is_dir():
        for item in path.iterdir():
            recursive_sha256(item, hashsum)
    else:
        hashsum.update(path.read_bytes())

    return str(hashsum.hexdigest())
