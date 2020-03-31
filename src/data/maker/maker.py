from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, Mapping, Sized, TypeVar


class Maker(ABC, Sized):
    def __init__(self, raw_path: Path, out_path: Path):
        self.raw_path = raw_path
        self.out_path = out_path

    @property
    @abstractmethod
    def out_file(self) -> Path:
        pass

    @property
    @abstractmethod
    def hash(self) -> str:
        pass

    @abstractmethod
    def make(self) -> None:
        pass

    @abstractmethod
    def pre(self) -> bool:
        pass

    @abstractmethod
    def post(self) -> None:
        pass

    def convert(self) -> None:
        self.pre()
        self.make()
        self.post()
