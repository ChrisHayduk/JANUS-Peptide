import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'alphafold-parallel-msa'))

from alphafold.common import protein
from alphafold.common import residue_constants

def fitness_function(peptide: str, receptor_if_residues: str, feature_dict: dict, prediction_result: dict) -> float:
    #Calculate loss
    #Loss features
    # Get the pLDDT confidence metric.
    plddt = prediction_result['plddt']
    #Get the protein
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=feature_dict,result=prediction_result,b_factors=plddt_b_factors)

    protein_resno, protein_atoms, protein_atom_coords = protein.get_coords(unrelaxed_protein)
    peptide_length = len(peptide)
    #Get residue index
    residue_index = feature_dict['residue_index']
    receptor_res_index = residue_index[:-peptide_length]
    peptide_res_index = residue_index[-peptide_length:]
    #Get coords
    receptor_coords = protein_atom_coords[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
    peptide_coords = protein_atom_coords[np.argwhere(protein_resno>receptor_res_index[-1]+1)[:,0]]
    #Get atom types
    receptor_atoms = protein_atoms[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
    peptide_atoms = protein_atoms[np.argwhere(protein_resno>receptor_res_index[-1]+1)[:,0]]
    #Get resno for each atom
    #Start at 1 - same for receptor_if_residues
    receptor_resno = protein_resno[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
    peptide_resno = protein_resno[np.argwhere(protein_resno>peptide_res_index[0])[:,0]]
    #Get atoms belonging to if res for the receptor
    receptor_if_pos = []
    for ifr in receptor_if_residues:
        receptor_if_pos.extend([*np.argwhere(receptor_resno==ifr)])
    receptor_if_pos = np.array(receptor_if_pos)[:,0]

    #Calc 2-norm - distance between peptide and interface
    mat = np.append(peptide_coords,receptor_coords[receptor_if_pos],axis=0)
    a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(peptide_coords)
    #Get interface
    contact_dists = dists[:l1,l1:] #first dimension = peptide, second = receptor

    #Get the closest atom-atom distances across the receptor interface residues.
    closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]

    #Get the peptide plDDT
    peptide_plDDT = plddt[-peptide_length:]

    if_dist_peptide = closest_dists_peptide.mean()
    plddt = peptide_plDDT.mean()

    loss = if_dist_peptide*1/plddt

    return loss, if_dist_peptide, plddt, unrelaxed_protein