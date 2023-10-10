import json
from typing import Dict, Union, List
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt


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
        self.actions = set(gait["label"] for gait in self.data)

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

    def extract_sequences(self) -> List:
        """Returns a list of all the sequences in an accumulative way."""
        sequences = []
        for index, _ in enumerate(self.average_data_norm()):
            sequences.append(self.average_data_norm()[0 : index + 1])  # noqa : E203
        return sequences

    def compute_correlation(self) -> Dict[str, Dict[str, int]]:
        """Computes the correlation between different actions and
        prints the results in the console.
        """
        correlation = {}

        for index in range(len(self.data) - 1):
            current_action = self.data[index]["label"]
            next_action = self.data[index + 1]["label"]

            if current_action not in list(correlation.keys()):
                correlation[current_action] = {}

            if next_action not in correlation[current_action].keys():
                correlation[current_action][next_action] = 1
            else:
                correlation[current_action][next_action] += 1
        return correlation

    def plot_correlation_per_action(self, action: str) -> None:
        """Plot the correlation of a given action against all present actions.

        Args:
            action: given action.
        """
        correlated_actions = self.compute_correlation()[action]
        next_action_labels = list(correlated_actions.keys())
        next_action_counts = list(correlated_actions.values())

        plt.figure(figsize=(10, 5))
        plt.bar(next_action_labels, next_action_counts)
        plt.title(f"Actions following {action}")
        plt.xlabel("Next Actions")
        plt.ylabel("Count")
        plt.show()

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

    @property
    def average_gait_length_per_action(self) -> Dict[str, int]:
        """Compute the average length of the gait based on the action."""
        return {
            action: int(
                np.mean(
                    [
                        len(element["norm"])
                        for element in self.data
                        if element["label"] == action
                    ]
                )
            )
            for action in self.actions
        }
