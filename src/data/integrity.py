"""
File: integrity.py
Author: Orcan Tiryakioglu
Email: orcan.tiryakioglu@gmail.com
Github: https://github.com/nacro90
Description: Checks the integrity of the data using hashes
"""

import hashlib
from pathlib import Path
from typing import Any


DEFAULT_ENCODING = "UTF-8"

def recursive_sha256(path: Path, hashsum: Any = None) -> str:
    """
    Calculates sha256 hash of the file contents recursively.

    Args:
        path (Path): Parent path of contents
        hashsum (Optional[hashlib._HASH]): Current checksum of files if any

    Returns:
        str: Accumulated digest hex number string with lowercase letters like
            "03e93aae89012a2b06b77d5684f34bc2e27cd64e42108175338f20bec11c770a"

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


def str_sha256(s: str) -> str:
    """
    Calculates sha256 hash of the file contents recursively.

    Args:
        s (str): String value to check

    Returns:
        str: Accumulated digest hex number string with lowercase letters like
            "03e93aae89012a2b06b77d5684f34bc2e27cd64e42108175338f20bec11c770a"
    """
    hashsum = hashlib.sha256()
    hashsum.update(s.encode(DEFAULT_ENCODING))
    return hashsum.hexdigest()
