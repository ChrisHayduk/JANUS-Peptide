import yaml
from typing import List, Optional, Tuple
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import numpy as np

def validate_sequence(sequence: str, valid_amino_acids: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Validate an amino acid sequence
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence to validate
    valid_amino_acids : Optional[List[str]]
        List of valid amino acid codes. If None, uses standard 20 amino acids
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, cleaned_sequence)
    """
    if valid_amino_acids is None:
        valid_amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
    
    if sequence == '':
        return False, ''
        
    try:
        # Clean sequence
        sequence = ''.join(sequence.strip().upper().split())
        # Check if all characters are valid amino acids
        is_valid = all(aa in valid_amino_acids for aa in sequence)
        return is_valid, sequence
    except:
        return False, ''

def get_sequence_similarity(sequences: List[str], target_seq: str) -> List[float]:
    """
    Calculate similarity scores between a list of sequences and a target sequence
    using BLOSUM62 matrix and global alignment.
    
    Parameters
    ----------
    sequences : List[str]
        List of amino acid sequences to compare
    target_seq : str
        Target sequence to compare against
        
    Returns
    -------
    List[float]
        List of similarity scores between 0 and 1
    """
    similarity_scores = []
    blosum62 = matlist.blosum62
    
    for seq in sequences:
        # Perform global alignment with BLOSUM62 matrix
        alignment_score = pairwise2.align.globaldx(
            seq,
            target_seq,
            blosum62,
            score_only=True
        )
        
        # Normalize score
        max_possible_score = max(len(seq), len(target_seq)) * 4  # 4 is max BLOSUM62 score
        min_possible_score = max(len(seq), len(target_seq)) * (-4)  # -4 is min BLOSUM62 score
        score_range = max_possible_score - min_possible_score
        
        normalized_score = (alignment_score - min_possible_score) / score_range
        similarity_scores.append(normalized_score)
    
    return similarity_scores

def get_pairwise_similarity(seq1: str, seq2: str) -> float:
    """
    Calculate pairwise similarity between two sequences using BLOSUM62.
    
    Parameters
    ----------
    seq1 : str
        First amino acid sequence
    seq2 : str
        Second amino acid sequence
        
    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    return get_sequence_similarity([seq1], seq2)[0]

def get_sequence_properties(sequence: str) -> dict:
    """
    Calculate various properties of an amino acid sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
        
    Returns
    -------
    dict
        Dictionary containing sequence properties
    """
    # Amino acid property groups
    hydrophobic = set('AILMFWV')
    polar = set('QNSTC')
    charged = set('DEKRH')
    
    # Calculate percentages
    seq_len = len(sequence)
    hydrophobic_percent = sum(aa in hydrophobic for aa in sequence) / seq_len * 100
    polar_percent = sum(aa in polar for aa in sequence) / seq_len * 100
    charged_percent = sum(aa in charged for aa in sequence) / seq_len * 100
    
    return {
        'length': seq_len,
        'hydrophobic_percent': hydrophobic_percent,
        'polar_percent': polar_percent,
        'charged_percent': charged_percent
    }

def get_sequence_distance_matrix(sequences: List[str]) -> np.ndarray:
    """
    Calculate distance matrix for a list of sequences.
    
    Parameters
    ----------
    sequences : List[str]
        List of amino acid sequences
        
    Returns
    -------
    np.ndarray
        Distance matrix of sequence similarities
    """
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = get_pairwise_similarity(sequences[i], sequences[j])
            distance = 1 - similarity
            distance_matrix[i,j] = distance
            distance_matrix[j,i] = distance
            
    return distance_matrix

def from_yaml(work_dir: str, 
             fitness_function,
             start_population: str,
             yaml_file: str,
             **kwargs) -> dict:
    """
    Load parameters from YAML file and update with provided values.
    
    Parameters
    ----------
    work_dir : str
        Working directory path
    fitness_function : callable
        Function to evaluate sequence fitness
    start_population : str
        Path to file containing starting population
    yaml_file : str
        Path to YAML configuration file
    **kwargs : dict
        Additional parameters to override YAML values
        
    Returns
    -------
    dict
        Combined parameters dictionary
    """
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params.update(kwargs)
    params.update({
        'work_dir': work_dir,
        'fitness_function': fitness_function,
        'start_population': start_population
    })

    return params