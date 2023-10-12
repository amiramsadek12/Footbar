import json
import os
from typing import Dict, Union, List, Set
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

from src.utils import save_to_json


class Match:
    """Class holding the data of each match."""

    def __init__(self, path: str) -> None:
        """Initialize an instance of Match.

        Args:
            path:  path of match data JSON file.
        """
        if path:
            match_file = open(path)
            self.data: List[
                Dict[str, Union[str, float, List[float]]]
            ] = json.load(  # noqa:E501
                fp=match_file
            )
            self.path = path
            # Drop "no action" as it only figures twice in the second match
            for index, element in enumerate(self.data):
                if element["label"] == "no action":
                    self.data.pop(index)

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
        """Returns a list of all the sequences composed of all possible
        actions to be performed in a match in an accumulative way and only
        allow for an action to appear 3 times before another action comes
        in and only keep the latest 20 actions performed by the player."""
        sequences = []
        for index, _ in enumerate(self.average_data_norm()):
            sequences.append(self.average_data_norm()[0 : index + 1])  # noqa : E203

        sequences_with_only_actions = []
        for element in sequences:
            sequences_with_only_actions.append(
                [nested_element["label"] for nested_element in element]
            )

        new_sequences = [sequences_with_only_actions[0]]

        for sequence in sequences_with_only_actions:
            new_sequence = [sequence[0]]
            for action in sequence:
                consecutive_count = 1  # Counter for consecutive appearances
                max_consecutive = 3  # Maximum allowed consecutive appearances

                for action in sequence[1:]:
                    if action != new_sequence[-1]:
                        # Check if the action is different
                        # from the previous one
                        new_sequence.append(action)
                        consecutive_count = 1  # Reset the consecutive count
                    elif consecutive_count < max_consecutive:
                        new_sequence.append(action)
                        consecutive_count += 1
            new_sequences.append(new_sequence[-20:])

        return new_sequences

    def clean_data(self) -> List[List[str]]:
        """Remove all sequences that where only
        'walk' and 'run' are present."""
        all_sequences = self.extract_sequences()
        element_indices_to_be_deleted = []
        for index, sequence in enumerate(all_sequences):
            if all(item in ["walk", "run"] for item in sequence):
                element_indices_to_be_deleted.append(index)
        for index in element_indices_to_be_deleted:
            all_sequences.pop(index)

        return all_sequences

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

    def __add__(self, other):
        if isinstance(other, Match):
            dummy_path = "data.dummy_match.json"
            combined_sequences = self.data + other.data
            save_to_json(object_to_save=combined_sequences, path=dummy_path)
            dummy_match = Match(dummy_path)
            os.remove(dummy_path)
            return dummy_match
        else:
            raise TypeError("Unsupported operand type for +")

    @property
    def actions(self) -> Set:
        """Returns the actions performed in a match."""
        return set(gait["label"] for gait in self.data)

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

    def to_json(self) -> Dict:
        return self.data

    @classmethod
    def from_data(cls, data):
        instance = cls(None)
        instance.data = data
        return instance
