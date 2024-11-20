#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crossover operations for amino acid sequences
"""
import numpy as np
import random
from typing import List
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from .utils import get_sequence_similarity

def get_joint_similarity(all_sequences: List[str], starting_seq: str, target_seq: str) -> List[float]:
    """
    Get joint similarity values for all sequences, calculated with 
    reference to starting_seq & target_seq.

    Parameters
    ----------
    all_sequences : List[str]
        List of amino acid sequences
    starting_seq : str
        Input sequence
    target_seq : str
        Target sequence

    Returns
    -------
    List[float]
        List of joint similarity scores for all sequences
    """
    scores_start = get_sequence_similarity(all_sequences, starting_seq)
    scores_target = get_sequence_similarity(all_sequences, target_seq)
    data = np.array([scores_target, scores_start])

    avg_score = np.average(data, axis=0)
    better_score = avg_score - (np.abs(data[0] - data[1]))
    better_score = (
        ((1 / 9) * better_score ** 3)
        - ((7 / 9) * better_score ** 2)
        + ((19 / 12) * better_score)
    )

    return better_score

def obtain_path(starting_seq: str, target_seq: str) -> List[str]:
    """
    Create a single path between starting_seq and target_seq by gradually
    mutating one position at a time.

    Parameters
    ----------
    starting_seq : str
        Starting amino acid sequence
    target_seq : str
        Target amino acid sequence

    Returns
    -------
    List[str]
        List of sequences representing the path from start to target
    """
    # Convert sequences to lists for easier manipulation
    start_list = list(starting_seq)
    target_list = list(target_seq)

    # Pad the smaller sequence with gaps
    if len(start_list) < len(target_list):
        start_list.extend(['-'] * (len(target_list) - len(start_list)))
    else:
        target_list.extend(['-'] * (len(start_list) - len(target_list)))

    # Find positions where sequences differ
    indices_diff = [
        i
        for i in range(len(start_list))
        if start_list[i] != target_list[i]
    ]

    # Create path by mutating one position at a time
    path = {0: start_list.copy()}
    
    for iter_ in range(len(indices_diff)):
        idx = np.random.choice(indices_diff, 1)[0]  # Choose random position to mutate
        indices_diff.remove(idx)

        path_member = path[iter_].copy()
        path_member[idx] = target_list[idx]  # Mutate to target amino acid
        path[iter_ + 1] = path_member.copy()

    # Convert lists back to sequences, removing gaps
    path_sequences = []
    for i in range(len(path)):
        seq = ''.join(x for x in path[i] if x != '-')
        path_sequences.append(seq)

    return path_sequences

def perform_crossover(combined_seq: str, num_random_samples: int) -> List[str]:
    """
    Create multiple paths between sequences to obtain intermediate sequences
    representing crossover structures.

    Parameters
    ----------
    combined_seq : str
        Two sequences concatenated using xxx (example: MGKHDLxxxPAVKDLF)
    num_random_samples : int
        Number of different sequence orientations to consider

    Returns
    -------
    List[str]
        List of unique intermediate sequences encountered during path formation
    """
    seq_a, seq_b = combined_seq.split("xxx")
    
    # Generate sequence variations by randomly inserting gaps
    def generate_sequence_variations(seq: str, num_samples: int) -> List[str]:
        variations = []
        for _ in range(num_samples):
            seq_list = list(seq)
            # Randomly insert 0-3 gaps at random positions
            num_gaps = random.randint(0, 3)
            for _ in range(num_gaps):
                pos = random.randint(0, len(seq_list))
                seq_list.insert(pos, '-')
            variations.append(''.join(seq_list))
        return variations

    # Generate variations for both sequences
    variations_a = generate_sequence_variations(seq_a, num_random_samples)
    variations_b = generate_sequence_variations(seq_b, num_random_samples)

    # Generate paths between all variations
    collected_sequences = []
    for seq_1 in variations_a:
        for seq_2 in variations_b:
            path_sequences = obtain_path(seq_1, seq_2)
            collected_sequences.extend(path_sequences)

    # Remove gaps and duplicates
    clean_sequences = []
    for seq in collected_sequences:
        clean_seq = seq.replace('-', '')
        if clean_seq:  # Only add non-empty sequences
            clean_sequences.append(clean_seq)

    return list(set(clean_sequences))

def crossover_sequences(sequences_join: str, crossover_num_random_samples: int) -> List[str]:
    """
    Return a list of sequences (crossover results) ordered by joint similarity scores.

    Parameters
    ----------
    sequences_join : str
        Two sequences concatenated using xxx (example: MGKHDLxxxPAVKDLF)
    crossover_num_random_samples : int
        Number of random samples to generate for each sequence

    Returns
    -------
    List[str]
        List of crossover sequences ordered by joint similarity scores
    """
    map_ = {}
    map_[sequences_join] = perform_crossover(
        sequences_join, 
        num_random_samples=crossover_num_random_samples
    )

    for key_ in map_:
        med_all = map_[key_]
        seq_1, seq_2 = key_.split("xxx")
        joint_sim = get_joint_similarity(med_all, seq_1, seq_2)

        joint_sim_ord = np.argsort(joint_sim)[::-1]
        med_all_ord = [med_all[i] for i in joint_sim_ord]

    return med_all_ord