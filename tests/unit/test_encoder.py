from src.encoder import LabelEncoder
from src.match import Match
import os
import pytest


@pytest.fixture
def test_encoder():
    test_match = Match("tests/unit/test_data/test_match_1.json")
    actions = [gait["label"] for gait in test_match.data]
    test_encoder = LabelEncoder()
    test_encoder.fit(actions)
    return test_encoder


def test_fit(test_encoder):
    assert test_encoder.class_to_index == {"walk": 1, "rest": 0}
    assert test_encoder.index_to_class == {1: "walk", 0: "rest"}
    assert test_encoder.classes == ["rest", "walk"]


def test_encode(test_encoder):
    assert test_encoder.encode(["walk", "rest"]) == [1, 0]


def test_decode(test_encoder):
    assert test_encoder.decode([1, 0]) == ["walk", "rest"]


def test_save(test_encoder):
    path = "tests/unit/test_data/test_encoder.json"
    test_encoder.save(path)
    assert os.path.exists(path)
    assert os.access(path, os.R_OK)


def test_load(test_encoder):
    path = "tests/unit/test_data/test_encoder.json"
    test_encoder.save(path)
    loaded_encoder = LabelEncoder.load(path)
    assert test_encoder.class_to_index == loaded_encoder.class_to_index
    assert test_encoder.index_to_class == loaded_encoder.index_to_class
    assert test_encoder.classes == loaded_encoder.classes
