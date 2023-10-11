from abc import ABC, abstractmethod


class Data(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train_test_split(self) -> None:
        pass

    @abstractmethod
    def transform(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
