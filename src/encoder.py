import json
import numpy as np
from typing import Dict, List


class LabelEncoder:
    """Encode given data to be feed to the model."""

    def __init__(self, class_to_index: Dict[str, int] = {}) -> None:
        """Create an instance of the encoder.

        Args:
            class_to_index: optional dictinory mapping between actions and id.
        """
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def fit(self, data: List[str]) -> None:
        """Fit the encoder to the given data."""
        if isinstance(data[0], str):
            classes = np.unique(data)
        else:
            classes = set(element for sequence in data for element in sequence)
        for index, class_ in enumerate(classes):
            self.class_to_index[class_] = index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def encode(self, data: List[str]) -> List[int]:
        """Encode the given data using the fitted mapping."""
        if data == []:
            return
        if isinstance(data[0], str):
            encoded = np.zeros(len(data), dtype=int)
            for index, item in enumerate(data):
                encoded[index] = self.class_to_index.get(item)
        if isinstance(data[0], list):
            encoded = []
            for element in data:
                encoded.append(self.encode(element))
        return list(encoded)

    def decode(self, data: List[int]) -> List[str]:
        """Decode the given encoded data."""
        classes = []
        for _, item in enumerate(data):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp: str) -> None:
        """Save the mapping to a given path."""
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        """Create an instance from a saved file."""
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

    def to_json(self):
        return self.class_to_index
