from janus.janus import JANUS
from Bio import PDB
from Bio.PDB.Polypeptide import three_to_one
from Bio import SeqIO

import random
import os

def extract_sequence_from_pdb(pdb_path):
    """
    Extract amino acid sequences from each chain of a PDB file.
    
    Args:
        pdb_path (str): Path to PDB file
        
    Returns:
        dict: Dictionary mapping chain_id -> amino acid sequence in one-letter code
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    model = structure[0]
    chain_sequences = {}

    for chain in model:
        # Filter residues to only standard amino acids
        residues = [res for res in chain if PDB.is_aa(res) and res.id[0] == ' ']
        try:
            chain_sequence = ''.join([three_to_one(res.get_resname()) for res in residues])
            chain_sequences[chain.id] = chain_sequence
        except KeyError as e:
            print(f"Warning: Could not convert some residues in chain {chain.id}: {e}")
            chain_sequences[chain.id] = ""  # If conversion fails, set to empty or partial
    
    return chain_sequences

def extract_sequences_from_fasta(fasta_path):
    """
    Extract all sequences from a FASTA file.
    
    Args:
        fasta_path (str): Path to FASTA file
        
    Returns:
        OrderedDict: {record_id: sequence_str} for all sequences in the FASTA.
    """
    from collections import OrderedDict
    seqs = OrderedDict()
    for record in SeqIO.parse(fasta_path, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs

def generate_random_peptides(num_sequences=20, min_length=9, max_length=15, exclude_amino_acids=set()):
    """
    Generate random peptide sequences of varying lengths.
    
    Args:
        num_sequences (int): Number of sequences to generate
        min_length (int): Minimum peptide length
        max_length (int): Maximum peptide length
        
    Returns:
        list: List of random peptide sequences
    """
    # Standard amino acids in one-letter code
    amino_acids = set([
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ])
    
    amino_acids = amino_acids - exclude_amino_acids
    amino_acids = list(amino_acids)
    
    sequences = []
    for _ in range(num_sequences):
        # Random length between min and max
        length = random.randint(min_length, max_length)
        
        # Generate random sequence
        sequence = ''.join(random.choices(amino_acids, k=length))
        sequences.append(sequence)
    
    return sequences

# Define file paths
# Example usage:
# pdb_path = "/home/jupyter/input/monomerTfR1.pdb"
input_file = "/home/jupyter/input/tetramer.fasta"
output_file = "/home/jupyter/input/start_population.txt"

# Extract sequence(s) from FASTA file
fasta_sequences = extract_sequences_from_fasta(input_file)

# Convert the extracted dictionary into a list of chain sequences
# For example, if we have multiple chains in the FASTA:
# fasta_sequences might look like { "chain1": "ABC...", "chain2": "DEF..." }
# We'll just take all sequences and treat them as distinct chains.
# The order is based on input file (OrderedDict), but you can reorder if necessary.
target_sequence = list(fasta_sequences.values())

print(f"Extracted {len(target_sequence)} chain(s) from FASTA:")
for i, seq in enumerate(target_sequence, start=1):
    print(f"Chain {i}: {seq} (Length: {len(seq)})")

# Check if output file exists
if not os.path.exists(output_file):
    print(f"{output_file} does not exist. Generating random peptides.")
    # Generate random peptides
    sequences = generate_random_peptides(num_sequences=20, exclude_amino_acids=set(['C', 'M']))

    # Save to file
    with open(output_file, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")

    print(f"Generated {len(sequences)} sequences and saved to {output_file}:")
    for seq in sequences:
        print(f"{seq} (length: {len(seq)})")
else:
    print(f"{output_file} already exists. Skipping peptide generation.")

BUCKET_NAME = 'aln-lide-v1-cent-cvq-01-afbkt-01'
pipeline_name = 'multimer-pipeline-persistent-resource'

# Initialize the JANUS optimizer
janus = JANUS(
    # Required parameters
    work_dir="/home/jupyter/output/run1",                    # Directory to store results
    start_population="/home/jupyter/input/start_population.txt",        # File containing initial sequences
    target_sequence=target_sequence,    # A list of chain sequences (no peptide yet)
    project_id='test-ligand-design',    # Google Cloud project ID
    zone="us-central1-a",               # GCP region
    pipeline_root_path=f'gs://{BUCKET_NAME}/pipeline_runs/{pipeline_name}', 
    pipeline_template_path="/home/jupyter/JANUS-Peptide/src/multimer-pipeline-persistent-resource.json",
    pipeline_name=pipeline_name,        
    bucket_name=BUCKET_NAME,            
    experiment_id="peptide-opt-1",      
    verbose_out=True,                   
    generations=1000,                   
    generation_size=20,                 
    num_exchanges=5,                    
    explr_num_mutations=5,              
    exploit_num_mutations=20,           
    top_seqs=1,                         
    alphabet=['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
    use_small_bfd="true",
    skip_msa="true",
    num_multimer_predictions_per_model=1,
    model_names=['model_5_multimer_v3'],
    max_template_date='2030-01-01'
)

# Run the optimization
janus.run()
