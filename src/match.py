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

    def average_data_norm(self) -> List[Dict[str, Union[str, float]]]:
        return [
            {
                "label": gait["label"],
                "norm": np.round(np.mean(gait["norm"]), decimals=2),
            }
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

    @property
    def mean_norm_per_action(self) -> Dict:
        """
        Returns a dictionary containing the different actions
        in a match along with their averaged norm.

        """
        return {
            action: self.mean_norm_for_each_action(action)
            for action in list((self.count_actions()).keys())
        }

    def extract_sequences(self) -> List:
        """Returns a list of all the sequences in an accumulative way."""
        sequences = []
        for index, _ in enumerate(self.average_data_norm()):
            sequences.append(self.average_data_norm()[0 : index + 1])
        return sequences
