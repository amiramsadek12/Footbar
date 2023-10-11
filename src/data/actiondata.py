import ast
from encoder import LabelEncoder

from data.data import Data
from typing import Tuple, Any

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


RANDOM_STATE = 1234
MAX_SEQUENCE_LENGTH = 20


class ActionData(Data):
    def __init__(self, sequences) -> None:
        self.sequences = sequences

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

    def transform(self, encoder: LabelEncoder) -> Tuple[Any]:
        X_train_transformed = encoder.encode(self.X_train)
        y_train_transformed = encoder.encode(self.y_train)

        X_val_transformed = encoder.encode(self.X_val)
        y_val_transformed = encoder.encode(self.y_val)

        X_test_transformed = encoder.encode(self.X_test)
        y_test_transformed = encoder.encode(self.y_test)
        return (
            X_train_transformed,
            y_train_transformed,
            X_val_transformed,
            y_val_transformed,
            X_test_transformed,
            y_test_transformed,
        )

    def pad_sequences(
        self, X_train_transformed, X_val_transformed, X_test_transformed
    ) -> Tuple[Any]:
        X_train_padded = pad_sequences(
            X_train_transformed,
            padding="pre",
            truncating="pre",
            maxlen=MAX_SEQUENCE_LENGTH,
            value=18,
        )
        X_val_padded = pad_sequences(
            X_val_transformed,
            padding="pre",
            truncating="pre",
            maxlen=MAX_SEQUENCE_LENGTH,
            value=18,
        )
        X_test_padded = pad_sequences(
            X_test_transformed,
            padding="pre",
            truncating="pre",
            maxlen=MAX_SEQUENCE_LENGTH,
            value=18,
        )
        return X_train_padded, X_val_padded, X_test_padded

    def info(self) -> None:
        print(
            f"Size of training set: {len(self.X_train)} \n"
            f"Size of validation set: {len(self.X_val)} \n"
            f"Size of test set: {len(self.X_test)}"
        )

    def __add__(self, other):
        if isinstance(other, ActionData):
            combined_sequences = self.sequences + other.sequences
            return ActionData(combined_sequences)
        else:
            raise TypeError("Unsupported operand type for +")

    def save(self, fp) -> None:
        with open(fp, "w") as file:
            for item in self.sequences:
                file.write(str(item) + "\n")

    @classmethod
    def load(cls, fp) -> None:
        all_data = []
        with open(fp, "r") as file:
            for line in file:
                all_data.append(ast.literal_eval(line))
        return cls(all_data)
