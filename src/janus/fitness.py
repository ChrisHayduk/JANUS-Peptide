import numpy as np
import sys
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add the custom alphafold-parallel-msa directory to sys.path
custom_path = os.path.join(os.getcwd(), 'alphafold-parallel-msa', 'alphafold', 'common')
sys.path.insert(0, custom_path)

from protein import get_coords, from_prediction
from residue_constants import atom_type_num

def fitness_function(peptide: str, receptor_if_residues: str, feature_dict: dict, prediction_result: dict) -> Tuple[float, float, float, Any]:
    """
    Calculate fitness with comprehensive error handling.
    Returns: (loss, interface_distance, plddt, unrelaxed_protein)
    """
    def fail_fitness(unrelaxed_protein: Optional[Any] = None, error_msg: str = "") -> Tuple[float, float, float, Any]:
        """Enhanced failure reporting"""
        if error_msg:
            logger.error(f"Fitness calculation failed: {error_msg}")
        return float('-inf'), float('inf'), float('-inf'), unrelaxed_protein

    # Input validation
    if not isinstance(peptide, str) or not peptide:
        return fail_fitness(None, "Invalid or empty peptide sequence")
    
    if not isinstance(feature_dict, dict) or not isinstance(prediction_result, dict):
        return fail_fitness(None, "Invalid feature_dict or prediction_result type")

    # Validate required keys with type checking
    required_feature_keys = ['aatype', 'residue_index']
    for k in required_feature_keys:
        if k not in feature_dict:
            return fail_fitness(None, f"feature_dict missing key '{k}'")
        if not isinstance(feature_dict[k], np.ndarray):
            try:
                feature_dict[k] = np.array(feature_dict[k])
            except Exception as e:
                return fail_fitness(None, f"Could not convert feature_dict['{k}'] to numpy array: {e}")

    if 'plddt' not in prediction_result:
        return fail_fitness(None, "prediction_result missing 'plddt'")

    # Safe array conversions
    try:
        plddt = np.array(prediction_result['plddt'], dtype=float)
        if plddt.size == 0:
            return fail_fitness(None, "Empty plddt array")
        if plddt.ndim != 1:
            plddt = plddt.flatten()
    except Exception as e:
        return fail_fitness(None, f"Error processing plddt: {e}")

    # Process receptor interface residues
    try:
        if receptor_if_residues:
            receptor_if_residues = np.array(receptor_if_residues, dtype=int)
            if receptor_if_residues.size == 0:
                return fail_fitness(None, "Empty receptor_if_residues array")
        else:
            logger.info('No target residues provided. Using entire receptor sequence...')
            receptor_if_residues = np.arange(feature_dict['aatype'].shape[0], dtype=int)
    except Exception as e:
        return fail_fitness(None, f"Error processing receptor_if_residues: {e}")

    # Safe b_factors construction
    try:
        plddt_b_factors = np.repeat(plddt[:, None], atom_type_num, axis=-1)
    except Exception as e:
        return fail_fitness(None, f"Error constructing plddt_b_factors: {e}")

    # Create unrelaxed protein with error handling
    try:
        unrelaxed_protein = from_prediction(
            features=feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=False
        )
    except Exception as e:
        return fail_fitness(None, f"Error constructing unrelaxed_protein: {e}")

    # Validate protein shapes with detailed error messages
    try:
        n_res = unrelaxed_protein.aatype.shape[0]
        if n_res <= 0:
            return fail_fitness(unrelaxed_protein, "Zero or negative number of residues")

        expected_shapes = {
            'atom_positions': (n_res,),
            'atom_mask': (n_res,),
            'b_factors': (n_res,)
        }

        for attr, expected_shape in expected_shapes.items():
            actual_shape = getattr(unrelaxed_protein, attr).shape
            if not actual_shape[:len(expected_shape)] == expected_shape:
                return fail_fitness(unrelaxed_protein, 
                    f"Shape mismatch for {attr}: expected {expected_shape}, got {actual_shape}")
    except Exception as e:
        return fail_fitness(unrelaxed_protein, f"Error validating protein shapes: {e}")

    # Safe coordinate extraction
    try:
        protein_resno, protein_atoms, protein_atom_coords = get_coords(unrelaxed_protein)
        logger.debug(f"Atom coordinates shape: {unrelaxed_protein.atom_positions.shape}")
        if any(arr.size == 0 for arr in [protein_resno, protein_atoms, protein_atom_coords]):
            return fail_fitness(unrelaxed_protein, "Empty coordinate arrays")
            
        peptide_length = len(peptide)
        n_res = unrelaxed_protein.aatype.shape[0]

        # Identify which residues correspond to the peptide:
        # Assuming that the peptide is appended after the receptor residues.
        peptide_res_start = n_res - peptide_length
        peptide_res_end = n_res
        peptide_res_indices = np.arange(peptide_res_start, peptide_res_end, dtype=int)


        # Get coordinates for receptor and peptide
        receptor_coords = protein_atom_coords[:-peptide_length]
        peptide_coords = protein_atom_coords[-peptide_length:]
        receptor_resno = protein_resno[:-peptide_length]
        peptide_resno = protein_resno[-peptide_length:]

        logger.debug(f"Total residues: {len(unrelaxed_protein.chain_index)}")
        logger.debug(f"Peptide length: {peptide_length}")
        logger.debug(f"Receptor coordinates shape: {receptor_coords.shape}")
        logger.debug(f"Peptide coordinates shape: {peptide_coords.shape}")
        logger.debug(f"Sample receptor coords:\n{receptor_coords[:5]}")
        logger.debug(f"Sample peptide coords:\n{peptide_coords[:5]}")
        
        # Verify we have valid coordinates
        if receptor_coords.size == 0 or peptide_coords.size == 0:
            return fail_fitness(unrelaxed_protein, "Empty coordinate arrays after splitting")

    except Exception as e:
        return fail_fitness(unrelaxed_protein, f"Error splitting coordinates: {e}")

    # Safe distance calculation
    try:
        # Get interface positions
        receptor_if_pos = []
        for ifr in receptor_if_residues:
            mask = (receptor_resno == ifr)
            if np.any(mask):
                receptor_if_pos.extend(np.where(mask)[0])
        
        if not receptor_if_pos:
            return fail_fitness(unrelaxed_protein, "No interface positions found")
        
        receptor_if_pos = np.array(receptor_if_pos)
        interface_coords = receptor_coords[receptor_if_pos]
        
        logger.debug(f"Number of interface positions: {len(interface_coords)}")
        
        # Compute only peptide-to-interface distances:
        diff = peptide_coords[:, np.newaxis, :] - interface_coords[np.newaxis, :, :]
        contact_dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # shape: (peptide_length, number_of_interface_positions)
        closest_dists_peptide = np.min(contact_dists, axis=1)
        
        if np.all(closest_dists_peptide == 0):
            return fail_fitness(unrelaxed_protein, "All distances are zero")
            
        logger.debug(f"Distance range: {np.min(closest_dists_peptide)} to {np.max(closest_dists_peptide)}")

    except Exception as e:
        return fail_fitness(unrelaxed_protein, f"Error in distance calculation: {e}")

    # Safe metric calculations
    try:
        peptide_plDDT = plddt[-peptide_length:]
        if peptide_plDDT.size != peptide_length:
            return fail_fitness(unrelaxed_protein, "pLDDT length mismatch")

        # Safe mean calculations with bounds checking
        if_dist_peptide = np.mean(closest_dists_peptide) if closest_dists_peptide.size > 0 else float('inf')
        plddt_score = np.mean(peptide_plDDT) if peptide_plDDT.size > 0 else float('-inf')

        # Validate metrics
        if not np.isfinite(if_dist_peptide) or not np.isfinite(plddt_score) or plddt_score <= 0:
            return fail_fitness(unrelaxed_protein, "Invalid metric values")

        # Safe loss calculation
        loss = if_dist_peptide * (1 / plddt_score)
        if not np.isfinite(loss):
            return fail_fitness(unrelaxed_protein, "Invalid loss value")
        
        fitness = 1/loss if loss != 0 else float('inf')

    except Exception as e:
        return fail_fitness(unrelaxed_protein, f"Error computing metrics: {e}")

    return float(fitness), float(if_dist_peptide), float(plddt_score), unrelaxed_protein