from src.match import Match
import pytest


@pytest.mark.parametrize(
    "path_to_test_match, expected_results",
    [
        ("tests/unit/test_data/test_match_1.json", {"walk": 3, "rest": 1}),
        (
            "tests/unit/test_data/test_match_2.json",
            {"walk": 4, "rest": 4, "run": 3, "dribble": 2},
        ),
    ],
)
def test_count_actions(path_to_test_match, expected_results):
    test_match = Match(path_to_test_match)
    assert test_match.count_actions() == expected_results


@pytest.mark.parametrize(
    "path_to_test_match, expected_results",
    [
        ("tests/unit/test_data/test_match_1.json", {"MIN": 2, "MAX": 6}),
        ("tests/unit/test_data/test_match_2.json", {"MIN": 1, "MAX": 14}),
    ],
)
def test_find_min_max(path_to_test_match, expected_results):
    test_match = Match(path_to_test_match)
    assert test_match.find_min_max() == expected_results


@pytest.mark.parametrize(
    "path_to_test_match, expected_results",
    [
        (
            "tests/unit/test_data/test_match_1.json",
            [["walk", "walk", "walk", "rest"]],
        ),  # ignore: E501
        # (
        #     "tests/unit/test_data/test_match_2.json",
        #     [
        #         ["walk", "walk", "walk", "rest", "rest",
        #           "rest", "run","run","run","dribble", "dribble"],
        #         ["walk", "rest"],
        #     ],
        # ),
    ],
)
def test_extract_sequences(path_to_test_match, expected_results):
    test_match = Match(path_to_test_match)
    assert test_match.extract_sequences() == expected_results
