#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for handling amino acid sequence fragments and validation
"""
from typing import List, Tuple, Optional
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def validate_sequence(seq: str, valid_amino_acids: Optional[List[str]] = None) -> Tuple[str, bool]:
    """
    Validate an amino acid sequence and return a cleaned version.
    
    Parameters:
    -----------
    seq : str
        Amino acid sequence to validate
    valid_amino_acids : Optional[List[str]]
        List of valid amino acid codes. If None, uses standard 20 amino acids
        
    Returns:
    --------
    tuple : (str, bool)
        Clean sequence and boolean indicating if sequence is valid
    """
    if valid_amino_acids is None:
        valid_amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
    
    try:
        # Remove any whitespace and convert to uppercase
        seq_clean = ''.join(seq.split()).upper()
        
        # Check if all characters are valid amino acids
        is_valid = all(aa in valid_amino_acids for aa in seq_clean)
        
        return (seq_clean, is_valid)
    except:
        return (None, False)


def get_fragments(seq: str, window_size: int) -> List[str]:
    """
    Create overlapping fragments from sequence with given window size.
    Remove duplicates and empty fragments.
    
    Parameters:
    -----------
    seq : str
        Input amino acid sequence
    window_size : int
        Size of the sliding window for creating fragments
        
    Returns:
    --------
    List[str]
        List of unique sequence fragments
    """
    if window_size > len(seq):
        return [seq]
    
    frags = []
    for i in range(len(seq) - window_size + 1):
        frag = seq[i:i + window_size]
        frags.append(frag)
        
    # Remove duplicates and empty fragments
    return list(filter(None, list(set(frags))))


def form_fragments(seq: str, window_sizes: Optional[List[int]] = None) -> List[str]:
    """
    Create fragments of varying sizes from an amino acid sequence.
    Returns a list of unique fragments.
    
    Parameters:
    -----------
    seq : str
        Input amino acid sequence
    window_sizes : Optional[List[int]]
        List of window sizes to use for fragmentation.
        If None, uses [3, 4, 5] as default
        
    Returns:
    --------
    List[str]
        List of unique sequence fragments
    """
    if window_sizes is None:
        window_sizes = [3, 4, 5]  # Default window sizes
        
    # First validate the sequence
    seq_clean, is_valid = validate_sequence(seq)
    if not is_valid:
        return []
    
    # Get fragments for each window size
    all_fragments = []
    for size in window_sizes:
        fragments = get_fragments(seq_clean, size)
        all_fragments.extend(fragments)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_fragments = []
    for frag in all_fragments:
        if frag not in seen:
            seen.add(frag)
            unique_fragments.append(frag)
    
    return unique_fragments


def get_motifs(seq: str, motif_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Extract known protein motifs from sequence.
    This is an additional functionality specific to amino acid sequences.
    
    Parameters:
    -----------
    seq : str
        Input amino acid sequence
    motif_patterns : Optional[List[str]]
        List of motif patterns to search for.
        If None, uses some common motifs as example
        
    Returns:
    --------
    List[str]
        List of found motifs
    """
    if motif_patterns is None:
        # Example motif patterns (can be expanded)
        motif_patterns = [
            'KR[A-Z]{2,4}K',  # Nuclear localization signal
            'N[^P][ST][^P]',  # N-glycosylation site
            'C[A-Z]{2}C',     # Zinc finger motif
            'RGD'             # Cell attachment sequence
        ]
    
    import re
    found_motifs = []
    
    seq_clean, is_valid = validate_sequence(seq)
    if not is_valid:
        return []
    
    for pattern in motif_patterns:
        matches = re.finditer(pattern, seq_clean)
        for match in matches:
            found_motifs.append(match.group())
    
    return list(set(found_motifs))