from typing import List
import random

def mutate_sequence(
    sequence: str,
    num_mutations: int,
    valid_amino_acids: List[str] = None,
    min_length: int = 9,
    max_length: int = 15
) -> List[str]:
    """
    Given an input amino acid sequence, perform random mutations including substitutions,
    insertions, and deletions.

    Parameters
    ----------
    sequence : str
        Input amino acid sequence (e.g., "MGKHDL")
    num_mutations : int
        Number of mutations to perform on the sequence
    valid_amino_acids : List[str], optional
        List of valid amino acids to use for mutations. If None, uses standard 20 amino acids.

    Returns
    -------
    List[str]
        List of unique mutated sequences
    """
    if valid_amino_acids is None:
        valid_amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]

    def apply_mutation(seq: str) -> str:
        seq_list = list(seq)
        random_pos = random.choice(range(len(seq_list)))
        mutation_type = None

        if len(seq_list) == min_length:
            mutation_type = random.choice([1, 2])
        elif len(seq_list) == max_length:
            mutation_type = random.choice([1, 3])
        else:
            mutation_type = random.choice([1, 2, 3])

        if mutation_type == 1:  # substitution
            seq_list[random_pos] = random.choice(valid_amino_acids)
        elif mutation_type == 2:   # insertion
            seq_list.insert(random_pos, random.choice(valid_amino_acids))
        elif mutation_type == 3 and len(seq_list) > 1:  # deletion
            seq_list.pop(random_pos)
            
        return ''.join(seq_list)

    # Generate mutated sequences
    mutated_sequences = []
    current_seq = sequence
    
    for _ in range(num_mutations):
        current_seq = apply_mutation(current_seq)
        mutated_sequences.append(current_seq)

    return list(set(mutated_sequences))

if __name__ == "__main__":
    # Example usage
    sequences = [
        "MGKHKKLLF",
        "PAVKDLFMGKHDL",
        "MKKLLFPAVKDLF"
    ]
    
    for seq in sequences:
        mutated = mutate_sequence(seq, num_mutations=10)
        print(f"Original: {seq}")
        print(f"Mutated: {mutated}\n")