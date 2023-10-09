from typing import Any
import json


def save_to_json(object_to_save: Any, path: str) -> None:
    """Saves object in JSON file.

    Args:
        object_to_save: Given object to be saved.
        path: Path to the file hosting the object.
    """
    with open(path, "w") as file:
        json.dump(object_to_save, file)
