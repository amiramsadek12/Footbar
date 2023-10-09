import json
from typing import Dict, Union, List
from collections import Counter
import numpy as np


class Match:
    """Class holding the data of each match."""

    def __init__(self, path: str) -> None:
        """Initialize an instance of Match.

        Args:
            path:  path of match data JSON file.
        """
        match_file = open(path)
        self.data: List[Dict[str, Union[str, float, List[float]]]] = json.load(
            fp=match_file
        )

    def info(self) -> None:
        """Prints the number of actions in the match."""
        print(f"Number of gaits in this match is: {len(self.data)}")

    def count_actions(self) -> Dict[str, int]:
        """Returns the different actions performed in a
        match with their number of occurences."""
        return Counter(gait["label"] for gait in self.data)

    def find_min_max(self) -> Dict[str, int]:
        """Finds the minimum and maximum of the
        accelerometer number of dimensions.

        Args:
            match_data: Object containing a JSON document.
        """
        norm_lengths = [len(gait["norm"]) for gait in self.data]
        return {"MIN": min(norm_lengths), "MAX": max(norm_lengths)}

    def extract_sequences(self) -> List[Union[List, str]]:
        """Returns the different sequences of actions performed by a player.

        Args:
            match_data : JSON file containing match information.
        """
        actions = [gait["label"] for gait in self.data]
        first_action = actions[0]
        sequences = []
        sequence = [first_action]

        for index, action in enumerate(actions[1:]):
            if action != first_action and (
                index != len(actions) or actions[index + 1] == first_action
            ):
                sequence.append(action)
                sequences.append(sequence)
                first_action = actions[index + 1]
                sequence = [first_action]

            else:
                sequence.append(action)

        return sequences

    def average_data_norm(self) -> List[Dict[str, Union[str, float]]]:
        return [
            {"label": gait["label"], "norm": np.mean(gait["norm"])}
            for gait in self.data
        ]

    def mean_norm_for_each_action(self, action: str) -> float:
        """Returns the avearge norm of the actions.

        Args:
            match_data_averaged : JSON file containing match information
              with average norm.
        """

        return np.mean(
            [
                gait["norm"]
                for gait in self.average_data_norm()
                if gait["label"] == action
            ]
        )
