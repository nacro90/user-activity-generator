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
        print("\nConverting from raw dataset format to interim\n")
        print("Interim path: {}\nHash: {}".format(self.out_path, self.hash))
        if self.pre():
            print("Dataset already processed, skipping this step...")
            return
        print("Starting to convert. This can take a while...")
        self.make()
        print("Conversion completed. Dumping to converted data to {}".format(self.out_file.absolute()))
        self.post()
        print("Dump finished. Created interim successfull")
