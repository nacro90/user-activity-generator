import tempfile
import unittest

from src.data.integrity import *


class SingleFileIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        content = "<<test content>>"

        self.temp_file_path = tempfile.mkstemp()[1]
        with open(self.temp_file_path, "w") as f:
            f.write(content)

    def test_single_file(self) -> None:
        result = recursive_sha256(Path(self.temp_file_path))
        expected = "9c94978287e47ca36d1f9b78afeae1eb84889b56a9c5931394e01ef07e1d43b2"
        self.assertEqual(result, expected, "Hex-digests are different from each other")

    def tearDown(self) -> None:
        remove_recursively(Path(self.temp_file_path))


class FileTreeIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        content_a = "<<test content A>>"
        content_b = "<<test content B>>"

        self.temp_parent_path = tempfile.mkdtemp()
        temp_file_a_path = tempfile.mkstemp(dir=self.temp_parent_path)[1]
        temp_inner_dir_path = tempfile.mkdtemp(dir=self.temp_parent_path)
        temp_file_b_path = tempfile.mkstemp(dir=temp_inner_dir_path)[1]

        with open(temp_file_a_path, "w") as f:
            f.write(content_a)

        with open(temp_file_b_path, "w") as f:
            f.write(content_b)

    def test_file_tree(self) -> None:
        result = recursive_sha256(Path(self.temp_parent_path))
        expected = "36632eab36489f5320d1b8ee0c2979fbdc5c18ccafe1264b208804f52a2d8710"
        self.assertEqual(result, expected, "Hex-digests are different from each other")

    def tearDown(self) -> None:
        remove_recursively(Path(self.temp_parent_path))


def remove_recursively(path: Path) -> None:
    if path.is_dir():
        for node in path.iterdir():
            remove_recursively(node)
        path.rmdir()
    else:
        path.unlink()
