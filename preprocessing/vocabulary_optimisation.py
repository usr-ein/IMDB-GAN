from typing import Set, Dict

import numpy as np

from generate_dataset import add_padding


def word_freq(dataset: np.ndarray) -> Dict[str, int]:
    freqs = dict()
    for entry in dataset:
        for word in entry:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1
    return freqs


def get_top_common(dataset: np.ndarray, to_keep: float = 0.5) -> Set[str]:
    freqs = word_freq(dataset)
    sorted_freqs = sorted(freqs.items(), key=lambda i: -i[1])
    cutting_point = int(len(sorted_freqs) * to_keep)

    reduced_set = {word for word, freq in sorted_freqs[:cutting_point]}
    return reduced_set


def remove_uncommon(dataset: np.ndarray, to_keep: float = 0.5) -> (np.ndarray, Dict[str, int]):
    most_frequent = get_top_common(dataset, to_keep)

    # way too violent
    # Removes any line that contains at least one word not in the most frequent
    # tmp_dataset = [row for row in dataset if all(e in most_frequent for e in row)]

    # Removes words that are not in the most frequent
    tmp_dataset = [[e for e in row if e in most_frequent] for row in dataset]

    new_dataset = add_padding(tmp_dataset)

    return new_dataset


def remove_unused_vocabulary(dataset, vocab_dict):
    set_of_used_words = set(dataset.flatten().tolist())
    return {w: i for w, i in vocab_dict.items() if i in set_of_used_words}


def reindex_vocabulary(dataset, vocab_dict):
    transition_dict = {old_i: i for i, old_i in zip(range(len(vocab_dict), vocab_dict.keys()))}

    new_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            new_dataset = transition_dict[dataset[i, j]]

    new_dict = {w: transition_dict[old_i] for w, old_i in vocab_dict.items()}
    return new_dataset, new_dict
