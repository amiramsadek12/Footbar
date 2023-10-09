import os
from src.utils import save_to_json
import pytest


@pytest.mark.parametrize(
    "object_to_save, path",
    [
        ({"name": "foo"}, "tests/unit/test_data/foo.json"),
        ({"name": "bar"}, "tests/unit/test_data/bar.json"),
    ],
)
def test_save_to_json(object_to_save, path):
    save_to_json(object_to_save, path)
    assert os.path.exists(path)
    assert os.access(path, os.R_OK)
    os.remove(path)
