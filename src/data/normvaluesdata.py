import json

import numpy as np
from encoder import LabelEncoder
from match import Match
from data.data import Data
import joblib


from typing import Union, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1234


class NormValuesData(Data):
    def __init__(self, match: Union[Match, str], encoder: Union[LabelEncoder, str]):
        if isinstance(match, Match) and isinstance(encoder, LabelEncoder):
            self.match = match
            self.encoder = encoder
        if isinstance(match, str) and isinstance(encoder, str):
            # Create a match from a file holding only data.
            self.match = Match(match)
            self.encoder = LabelEncoder.load(encoder)

        self.X = [
            self.encoder.encode([element["label"]])
            for element in self.match.data
            for _ in range(len(element["norm"]))
        ]
        self.y = [[norm] for element in self.match.data for norm in element["norm"]]
        try:
            self.scaler = joblib.load("../data/model/scaler.pkl")
        except FileNotFoundError:
            self.scaler = StandardScaler()

    @property
    def action_interval(self):
        empty_sequences_count = 0
        result = {}
        for action in self.match.actions:
            try:
                encoded_action = self.encoder.encode([action])
                data = [
                    np.mean(element["norm"])
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
            print(f"Skipping {empty_sequences_count} sequences as they are empty.")
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

    def transform(self) -> Tuple[Any]:
        self.scaler.fit(self.y_train)
        joblib.dump(self.scaler, "../data/model/scaler.pkl")
        y_train_scaled = self.scaler.transform(self.y_train)
        y_val_scaled = self.scaler.transform(self.y_val)
        y_test_scaled = self.scaler.transform(self.y_test)
        return y_train_scaled, y_val_scaled, y_test_scaled

    def inverse_transform(
        self, y_train_scaled, y_val_scaled, y_test_scaled
    ) -> [Tuple[Any]]:
        y_train = self.scaler.inverse_transform(y_train_scaled)
        y_val = self.scaler.inverse_transform(y_val_scaled)
        y_test = self.scaler.inverse_transform(y_test_scaled)
        return y_train, y_val, y_test

    def item_inverse_transform(self, item) -> [Tuple[Any]]:
        return self.scaler.inverse_transform(item)

    def __add__(self, other):
        if isinstance(other, NormValuesData):
            combined_match = self.match + other.match
            return NormValuesData(combined_match, encoder=self.encoder)
        else:
            raise TypeError("Unsupported operand type for +")

    def save(self, fp="../data/gait_data/norm_data.json"):
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
    def load(cls, fp="../data/gait_data/norm_data.json"):
        """load the model."""
        with open(fp, "r") as file:
            kwargs = json.load(file)
        return cls(**kwargs)
