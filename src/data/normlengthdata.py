import json


from data.data import Data
from match import Match
from encoder import LabelEncoder

import numpy as np
from typing import Tuple, Any, Union


from sklearn.model_selection import train_test_split


RANDOM_STATE = 1234


class NormLengthData(Data):
    def __init__(
        self,
        match: Union[Match, str],
        encoder: Union[LabelEncoder, str] = None,  # noqa:E501
    ) -> None:
        if isinstance(match, Match) and isinstance(encoder, LabelEncoder):  # noqa:E501
            self.match = match
            self.encoder = encoder
        if isinstance(match, str) and isinstance(encoder, str):
            # Create a match from a file holding only data.
            self.match = Match(match)
            self.encoder = LabelEncoder.load(encoder)

        self.X = [[element["label"]] for element in self.match.data]
        self.y = [len(element["norm"]) for element in self.match.data]

    @property
    def action_interval(self):
        empty_sequences_count = 0
        result = {}
        for action in self.match.actions:
            try:
                encoded_action = self.encoder.encode([action])
                data = [
                    len(element["norm"])
                    for element in self.match.data
                    if element["label"] == action
                ]
                result[encoded_action[0]] = [
                    np.mean(data),
                    np.std(data),
                ]
            except TypeError:
                empty_sequences_count += 1
        if empty_sequences_count != 0:
            print(
                f"Skipping {empty_sequences_count} sequences as they are empty."  # noqa:E501
            )
        return result

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
            return NormLengthData(combined_match, encoder=self.encoder)
        else:
            raise TypeError("Unsupported operand type for +")

    def save(self, fp="../data/gait_data/gait_data.json"):
        """Save the data."""
        with open(fp, "w") as file:
            json.dump(
                {
                    "match": "../data/gait_data/match_data.json",
                    "encoder": "../data/gait_data/encoder.json",
                },
                file,
            )

    @classmethod
    def load(cls, fp="../data/gait_data/gait_data.json"):
        """load the model."""
        with open(fp, "r") as file:
            kwargs = json.load(file)
        return cls(**kwargs)
