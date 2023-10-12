import json
from data.data import Data
from match import Match
from encoder import LabelEncoder

from typing import Tuple, Any, Union


from sklearn.model_selection import train_test_split


RANDOM_STATE = 1234


class NormLengthData(Data):
    def __init__(self, match: Match) -> None:
        self.match = match
        self.X = [[element["label"]] for element in match.data]
        self.y = [len(element["norm"]) for element in match.data]

    def train_test_split(self, X, y, train_size=0.7) -> None:
        self.X_train, _X, self.y_train, _y = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=RANDOM_STATE,
            shuffle=True,  # noqa:E501
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            _X, _y, train_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )

    def info(self) -> None:
        print(
            f"Size of training set: {len(self.X_train)} \n"
            f"Size of validation set: {len(self.X_val)} \n"
            f"Size of test set: {len(self.X_test)}"
        )

    def transform(
        self, encoder: LabelEncoder = None
    ) -> Union[None, Tuple[Any]]:  # noqa:E501
        # There is no need to transform the inputs as they are
        # already encoded from the action model output.
        if encoder:
            X_train_transformed = encoder.encode(self.X_train)
            X_val_transformed = encoder.encode(self.X_val)
            X_test_transformed = encoder.encode(self.X_test)
            return (
                X_train_transformed,
                X_val_transformed,
                X_test_transformed,
            )
        return

    def __add__(self, other):
        if isinstance(other, NormLengthData):
            combined_match = self.match + other.match
            return NormLengthData(combined_match)
        else:
            raise TypeError("Unsupported operand type for +")

    def save(self, fp: str) -> None:
        with open(fp, "w") as fp:
            contents = {
                element[0]: element[1] for element in zip(self.X, self.y)
            }  # noqa:E501
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        """Create an instance from a saved file."""
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
