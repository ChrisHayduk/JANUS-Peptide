# JANUS-Peptide README

Welcome to **JANUS-Peptide**, a framework that wraps AlphaFold (Multimer) within a Genetic Algorithm (GA) to design and optimize novel peptide binders against a given target protein. This approach is built on concepts from the JANUS system—repurposed here to integrate structural feedback from protein predictions (via AlphaFold2 Multimer) into an iterative GA search procedure. The overarching goal is to evolve peptides that exhibit improved binding characteristics to a specified receptor or protein of interest.

This README will guide you through:

1. [High-level Purpose and Concepts](#high-level-purpose-and-concepts)  
2. [Core Functionality](#core-functionality)  
3. [Detailed File Descriptions and Code Summaries](#detailed-file-descriptions-and-code-summaries)  
   - [@run.py](#runpy)  
   - [@janus.py](#januspy)  
4. [Usage Instructions](#usage-instructions)  
   - [Basic Setup](#basic-setup)  
   - [Running the Optimization](#running-the-optimization)  
   - [Expected Outputs and Checkpoints](#expected-outputs-and-checkpoints)  
5. [Critical Dependencies](#critical-dependencies)  
6. [Best Practices and Notes](#best-practices-and-notes)  
   - [Using Multiple GPUs / Data Parallelism](#using-multiple-gpus--data-parallelism)  
   - [Checkpointing](#checkpointing)  
   - [Customizing the Genetic Algorithm Operators](#customizing-the-genetic-algorithm-operators)  

## High-level Purpose and Concepts

**JANUS-Peptide** allows you to automatically generate peptide sequences that are intended to bind to a target protein. Here is the workflow in simple terms:

1. **Initial Population**: You either provide or generate an initial set of candidate peptide sequences.  
2. **Genome Representation**: Each sequence is treated as an individual in a GA.  
3. **Fitness Computation**:  
   - Each candidate sequence is appended to your target protein’s coordinates in a single FASTA file (multimer format).  
   - AlphaFold2 Multimer is run in the cloud (or locally) to generate structural models of the target + candidate peptide complex.  
   - A custom fitness function then evaluates each complex (computing pLDDT, an interface distance metric, or any other structural/binding metric you supply).  
4. **Genetic Algorithm Operators**:  
   - **Mutation**: Randomly introduce single- or multi-point changes in the peptide sequences.  
   - **Crossover**: Recombine existing sequences to create new variants.  
5. **Selection & Replacement**: Based on fitness, the GA iteratively selects top candidates and replaces less-fit individuals with improved offspring.  
6. **Multiple Generations**: The process repeats for a user-specified number of generations until the peptides converge on highly fit solutions.

This pipeline aims to **reduce guesswork** by using a combination of data-driven local exploration (mutation) and broader structural exploration (crossover). The use of AlphaFold2 Multimer predictions ensures that you get immediate feedback on binding potential or structural plausibility of the designed peptide.

## Core Functionality

- **Automated Preparation** of target + peptide sequences for AlphaFold2 Multimer prediction.  
- **Remote Execution** on Google Cloud (Vertex AI Pipelines), handling HPC or GPU usage behind the scenes if needed.  
- **Iterative Genetic Algorithm**:
  - Takes user-defined mutation rates, crossover probabilities, and population sizes.  
  - Provides easy ways to plug in custom filtering and custom fitness functions.  
- **Checkpointing**:  
  - Saves intermediate states after each generation.  
  - Stores partial results, enabling you to resume from the latest checkpoint if an error occurs or if you need to interrupt execution.  
- **Logging & CSV Summaries**:  
  - Tracks population fitness logs, best sequences, and global histories.

## Detailed File Descriptions and Code Summaries

### @run.py

```python:src/janus/run.py
# Main entry point that demonstrates creating a JANUS instance and running the algorithm

def extract_sequence_from_pdb(pdb_path):
    ...
```

#### Purpose

- **`extract_sequence_from_pdb`**: Uses BioPython’s PDB parser to read a PDB file, filter out non-standard residues, and convert three-letter amino acid codes to one-letter codes.
- **`extract_sequences_from_fasta`**: Reads sequences from a multi-FASTA file and returns them as an ordered dictionary. Helpful if your target protein is a single or multi-chain system.
- **`generate_random_peptides`**: Creates a default set of random peptides of varying lengths to bootstrap your GA population if you do not already have an existing set of peptides.

## Updated Usage Outline

Below is an expanded overview of **how to run the optimization process** and **what each parameter does** in the `JANUS` class. When you instantiate `JANUS`, you’ll pass in a variety of parameters that control where results get written, how the genetic algorithm operates, and how AlphaFold Multimer jobs are executed. **This section discusses each parameter in detail.**

> **Note**: The following parameters are defined in the `JANUS.__init__` method (see [@janus.py](#januspy)). Understanding these parameters will help you configure the peptide optimization process precisely as you need.

---

### 1. **`work_dir`** (str)
- **Purpose**: The directory where all results, logs, and checkpoint files are stored.
- **Typical usage**: Provide an absolute or relative path (e.g., `"./results"` or `"/home/user/run1"`).
- **Example**: `work_dir="/home/jupyter/output/run1"`

### 2. **`start_population`** (str)
- **Purpose**: Path to a text file containing the **initial population** of peptide sequences (one sequence per line).  
- **Initial population constraints**:
  - Must contain at least as many sequences as `generation_size`.
  - Sequences should only use characters from the `alphabet` parameter (or the default 20 standard amino acids if `alphabet` is not overridden).
- **Example**: `start_population="/home/jupyter/input/start_population.txt"`

### 3. **`target_sequence`** (str or List[str])
- **Purpose**: The amino acid sequence(s) of the target protein.  
  - Usually provided as a list of strings if the target has multiple chains. 
  - The code will bundle these target chain sequences with each candidate peptide when creating the FASTA for AlphaFold Multimer.
- **Example**: `target_sequence=["MGGKWS...DIE"]` (single chain) or `target_sequence=["MGGKWS...DIE","TKQS...DFG"]` (multi-chain)

### 4. **`project_id`** (str)
- **Purpose**: The Google Cloud project ID under which Vertex AI Pipelines will be run (if using GCP as the execution backend).
- **Example**: `project_id="my-gcp-project"`

### 5. **`zone`** (str)
- **Purpose**: The GCP compute zone or location (e.g., `"us-central1-a"`) used to determine the `region` and schedule Vertex AI resources.
- **Note**: Internally, `JANUS` derives `self.region` from this parameter by removing the trailing “-a”, “-b”, etc.

### 6. **`pipeline_root_path`** (str)
- **Purpose**: The Google Cloud Storage path serving as the root for Vertex AI Pipeline runs/artifacts.
- **Example**: `pipeline_root_path="gs://my-bucket/pipeline_root"`

### 7. **`pipeline_template_path`** (str)
- **Purpose**: A path (local or GCS) to the JSON file describing the Vertex AI Pipeline template.  
- **Example**: `pipeline_template_path="/home/jupyter/src/multimer-pipeline.json"`

### 8. **`pipeline_name`** (str)
- **Purpose**: The display name (in Vertex AI) for the pipeline runs.  
- **Example**: `pipeline_name="multimer-pipeline-persistent-resource"`

### 9. **`bucket_name`** (str)
- **Purpose**: The name of the GCS bucket used for staging input sequences, storing AlphaFold results, and logging pipeline outputs.  
- **Note**: Do **not** include “gs://” prefix here; just the bucket name.
- **Example**: `bucket_name="my-af2-bucket"`

### 10. **`experiment_id`** (str)
- **Purpose**: A string identifier to tag all pipeline runs and sequence artifacts so that you can differentiate them from other experiments in the same bucket.
- **Example**: `experiment_id="peptide-opt-1"`

### 11. **`verbose_out`** (bool, default: `False`)
- **Purpose**: Controls whether the pipeline writes more verbose logs and data dumps (such as exploration-phase sequence logs, local search logs, etc.) into separate directories.
- **Example**:  
  - **`True`**: Creates generation-specific folders like `/run1/0_DATA/`, `/run1/1_DATA/`, etc. to store additional intermediate results.  
  - **`False`**: Writes fewer files, primarily top-level logs.

### 12. **`custom_filter`** (Callable or `None`, default: `None`)
- **Purpose**: A user-defined function (if needed) to filter out undesired sequences. You could, for example, disallow certain motifs or repeated amino acids.
- **Usage**: 
  - If `None`, no additional filter is applied. 
  - If provided, it should accept a list of sequences and return the filtered list of valid sequences.
  
### 13. **`alphabet`** (List[str] or `None`)
- **Purpose**: The set of valid amino acids to use when generating mutations or random peptides.  
- **Default**: The 20 standard amino acids (`[A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]`).  
- **Example**: `alphabet=['A','D','E','F','G']` to restrict to a tiny subset.

### 14. **`use_gpu`** (bool, default: `True`)
- **Purpose**: Whether to allocate a GPU when running AlphaFold predictions through Vertex AI or local HPC.  
- **Note**: Depending on your HPC or GCP environment setup, turning this off might default to CPU-only predictions (which are much slower).

### 15. **`num_workers`** (int or `None`, default: `None`)
- **Purpose**: The number of parallel workers to use for certain tasks.  
- **Current Behavior**: Although still present, the code does not heavily use multiprocessing. Typically set to `1` or left as `None`.

### 16. **`generations`** (int, default: `300`)
- **Purpose**: The total number of GA generations (optimization loops) to run.
- **Note**: Each generation can trigger new sequences to be evaluated, mutated, or replaced.

### 17. **`generation_size`** (int, default: `20`)
- **Purpose**: The size of the population in each generation—the number of sequences undergoing fitness evaluation and selection.

### 18. **`num_exchanges`** (int, default: `5`)
- **Purpose**: How many of the best local search sequences to replace in the main population during the exploitation phase.  
- **Typical Behavior**: At the end of a local search, the code picks the top `num_exchanges` sequences from local exploitation and replaces the worst `num_exchanges` sequences in the main population.

### 19. **`explr_num_random_samples`** (int, default: `5`)
- **Purpose**: In the mutation function, how many random attempts are made per sequence during the “exploration” phase (generating new candidate sequences).

### 20. **`explr_num_mutations`** (int, default: `5`)
- **Purpose**: Within the exploration phase, how many amino acid changes are introduced into an individual sequence each time it is mutated.

### 21. **`crossover_num_random_samples`** (int, default: `1`)
- **Purpose**: In the code base, this can control the number of crossovers attempted in certain contexts. Currently, the exploration phase typically pairs sequences for crossovers, mixing them with the best sequences.

### 22. **`exploit_num_random_samples`** (int, default: `20`)
- **Purpose**: Similar to `explr_num_random_samples` but used in the exploitation (local search) phase.  
- **Effect**: Each sequence can be duplicated multiple times for mutation or local improvement.

### 23. **`exploit_num_mutations`** (int, default: `20`)
- **Purpose**: The number of amino acid changes introduced into an individual sequence in the **local exploitation** phase. This is usually larger, as local search tries more aggressive mutations around top-ranked sequences.

### 24. **`top_seqs`** (int, default: `1`)
- **Purpose**: The number of top sequences carried forward directly without replacement (though the code also does this implicitly with selection processes).  
- **Note**: Must be <= `generation_size`.

### 25. **`use_small_bfd`** (str, default: `"true"`)
- **Purpose**: Tells the AlphaFold2 pipeline whether to use a smaller BFD database. Useful if your GCP setup has certain storage constraints or to speed up MSA generation (sometimes with a trade-off in performance).
- **Accepted Values**: `"true"` or `"false"`.

### 26. **`skip_msa`** (str, default: `"true"`)
- **Purpose**: Tells the pipeline to skip MSA (Multiple Sequence Alignment) generation for speed, if desired.  
- **Accepted Values**: `"true"` or `"false"`.

### 27. **`num_multimer_predictions_per_model`** (int, default: `1`)
- **Purpose**: How many times each AlphaFold model is run with different random seeds, for example, in multimer mode. Higher values can sometimes yield more diverse conformations but will increase compute time.

### 28. **`is_run_relax`** (str, default: `""`)
- **Purpose**: If set to a truthy string (e.g., `"true"`), the pipeline will attempt to relax the predicted structures with AMBER relaxation.  
- **Typically**: Left empty (`""`) or `"false"` if you do not want relaxation.

### 29. **`max_template_date`** (str, default: `'2030-01-01'`)
- **Purpose**: A date string specifying the maximum template release cutoff used by AlphaFold.  
- **Usage**: This is often set to a near-future date if your pipeline includes template-based predictions.

### 30. **`model_names`** (List[str], default: `['model_5_multimer_v3']`)
- **Purpose**: Which AlphaFold multimer model(s) to run. A list of model keys recognized by your pipeline.  
- **Example**: `model_names=['model_1_multimer_v3','model_2_multimer_v3']`

### 31. **`receptor_if_residues`** (str, default: `""`)
- **Purpose**: A string containing an optional specification of receptor interface residues or any annotated sites of interest. This can be used inside `fitness_function` to compute specialized metrics focusing on that region.
- **Example**: `"45-67,120-135"` if you want to focus on certain residue ranges in the target receptor.

---

### Example Instantiation

Below is a snippet showing how these parameters might be set up in your `run.py` script. Adjust them based on your setup and experimentation goals:

```python:src/janus/run.py
from janus.janus import JANUS

# Example usage:
janus = JANUS(
    # Required parameters
    work_dir="/home/jupyter/output/run1",          # Where to store results and logs
    start_population="/home/jupyter/input/start_population.txt",  # Contains initial peptide sequences
    target_sequence=["MGYKQ...", "YTHIF..."],      # Two-chain target protein
    project_id="my-gcp-project",
    zone="us-central1-a",
    pipeline_root_path="gs://my-bucket/pipeline_runs",
    pipeline_template_path="/home/jupyter/src/multimer-pipeline-persistent-resource.json",
    pipeline_name="multimer-pipeline-persistent-resource",
    bucket_name="my-bucket",
    experiment_id="peptide-opt-1",

    # Optional modifications
    verbose_out=True,           # True for more verbose file output
    generations=1000,           # More generations for a thorough search
    generation_size=20,         # Population size
    num_exchanges=5,            # Replace 5 worst sequences with best local sequences
    explr_num_mutations=5,      # Mutate 5 positions in exploration phase
    exploit_num_mutations=20,   # Mutate 20 positions in local exploitation
    top_seqs=1,
    alphabet=['A','D','E','F','G','H','I','K','L','N','P','Q','R','S','T','V','W','Y'],
    use_small_bfd="true",
    skip_msa="true",
    num_multimer_predictions_per_model=1,
    is_run_relax="false",
    max_template_date="2030-01-01",
    model_names=['model_5_multimer_v3'],
    receptor_if_residues=""     # Empty if no special interface restriction
)

# Run the optimization
janus.run()
```

### Summary of the Flow

1. **Setup**: You create a `JANUS` instance, pointing to your target sequences and either loading or generating an initial peptide population.  
2. **Configuration**: Adjust the parameters above to modify how many sequences each generation holds, how long to run the GA, how intensively to mutate or crossover, etc.  
3. **Execution**: `janus.run()` orchestrates the full GA loop, from exploration to exploitation. It also automatically manages checkpointing and CSV output for each generation.  
4. **Resulting Data**: Logs, CSV summaries, and predictive structures are stored in `work_dir` and in your GCS bucket (if you’re running on Vertex AI).

---

With these parameters, **you have full control over how the peptide design pipeline operates.** You can tune the mutation rates, number of generations, parallelization settings, and more to balance speed, exploration, and potential quality of solutions. Check the logs in your `work_dir` (and in Vertex AI Pipelines if using GCP) for progress updates, especially if you run many generations.

#### Purpose

Defines the core **JANUS** class, implementing all the core features of the GA:

1. **Constructor** (`__init__`)  
   Initializes GA hyperparameters, sets up logging, loads the initial sequence population, and performs basic input validation.  
2. **Fitness Calculation** (`compute_fitness_batch`)  
   - Submits each peptide + target combination to Vertex AI for AlphaFold2 Multimer inference.  
   - Waits for the predictions to finish and then parses results (pLDDT, interface distances, optional custom metrics).  
   - **Key**: This function is where the code fetches outputs from your cloud-based job.
3. **Mutation & Crossover**  
   - **`mutate_seq_list`**: Applies random changes to a set of sequences.  
   - **`crossover_seq_list`**: Pairs sequences for recombination.  
4. **Filter & Selection** (`check_filters`, `get_good_bad_sequences`)  
   - Optionally apply user-defined custom filters.  
   - Rank sequences by fitness and decide which ones to retain/replace.  
5. **Checkpointing** (`_save_checkpoint` & `_load_checkpoint`)  
   - Saves all relevant GA state (`population`, `fitness`, etc.).  
   - Restarts from the saved checkpoint if the job is interrupted or fails.  
6. **Fitness Scoring**  
   - Inside `_process_predictions`:  
     - Determines the best-scoring model out of multiple multimer predictions.  
     - Extracts relevant 3D metrics (confidence, interface distances).  
     - Passes them to the **`fitness_function`** (imported from `fitness.py`) for a single numeric summary of “fitness.”  
7. **GA Lifecycle** (`run`)  
   - Main loop over many generations.  
   - Each iteration:  
     1. Save checkpoint.  
     2. Perform exploration (mutation/crossover) to create new candidate peptides.  
     3. Compute fitness for new candidates.  
     4. Perform exploitation (more local intensification around best sequences).  
     5. Replace worst sequences with top solutions from local search.  
     6. Write generation summaries and global CSV logs.  

## Usage Instructions

### Basic Setup

1. **Clone or Download** the repository containing `run.py` and `janus.py`.  
2. **Install Dependencies** (see [Critical Dependencies](#critical-dependencies)).  
3. **Prepare Target Sequence**:  
   - You can have a single multi-chain FASTA, or multiple FASTA files, or a PDB structure from which you extract chain sequences.  

### Running the Optimization

1. **Edit `run.py`**:
   - Set `input_file` to your target’s FASTA file.  
   - (Optional) If you wish to auto-generate an initial peptide population, ensure `output_file` is defined (e.g., `start_population.txt`). If this file does not exist, `run.py` will generate random peptides.  
   - Update any GCP environment details—e.g., `BUCKET_NAME`, `pipeline_name`, or local HPC parameters if running in a different environment.  
2. **Execute**:
   ```bash
   python src/janus/run.py
   ```
3. **Monitoring**:
   - The script prints progress logs.  
   - Generation-level CSV files and logs appear in the `work_dir`.  
   - If running on GCP Vertex AI, you may monitor pipeline jobs in the [Vertex AI Pipelines Console](https://cloud.google.com/vertex-ai/docs/pipelines).  

### Expected Outputs and Checkpoints

- **`init_seqs.txt`**: The initial set of peptides used by the GA.  
- **`checkpoint.pkl`**: Updated at the start of each generation; stores the current population, fitnesses, etc.  
- **`generation_<N>_peptides.csv`**: Summaries of the population for each generation.  
- **`global_peptides.csv`**: Contains all sequences that have ever been evaluated, along with their best fitness.  
- **`<gen>_DATA/`** (if `verbose_out=True`): Additional intermediate logs, such as the raw fitness arrays for exploration and local searches.  

## Critical Dependencies

1. **Python 3.8+**  
2. **BioPython** (e.g., `pip install biopython`) – used for PDB parsing, FASTA parsing, pairwise alignments.  
3. **Google Cloud SDK** & **google-cloud-aiplatform** – for using Vertex AI if you’re running in a GCP environment.  
4. **NumPy**, **PyYAML**, **pickle** – for numeric arrays, hyperparameter storage, and serialization.  
5. **A suitable HPC or GPU environment** – If not on GCP, you should adapt the pipelines or calls to run locally or on your HPC cluster.  

## Best Practices and Notes

### Using Multiple GPUs / Data Parallelism

The code is designed so that individual AlphaFold2 multimer predictions can be launched in parallel to multiple Vertex AI Pipelines, each potentially using GPUs. If you wish to integrate local HPC usage:

- Use job scheduling software (e.g., SLURM) to parallelize the predictions.  
- Modify `compute_fitness_batch` to dispatch predictions to multiple GPUs, ensuring each job is pinned to a different device.  

Because each generation typically involves many sequences, you can speed up the process significantly by distributing the AlphaFold2 inference across multiple GPU nodes or tasks in parallel.

### Checkpointing

- **Automatic**: At each generation, the pipeline writes a `checkpoint.pkl` containing (1) the entire population, (2) their fitness values, (3) the global dictionary tracking all sequences.  
- **Resuming**: In the event of a crash, the code attempts to reload from the last checkpoint. Make sure the checkpoint file is present to continue from that state.

### Customizing the Genetic Algorithm Operators

- **Mutation**: Implement your own function in `mutate_sequence` or augment `mutate_seq_list` if you have specialized constraints (e.g., no more than two consecutive specific amino acids).  
- **Crossover**: To experiment with different recombination strategies for peptides, adjust or replace `crossover_sequences` in `crossover.py`.  
- **Fitness** (`fitness_function` in `fitness.py`): The default is a placeholder that calculates a combined scoring (pLDDT, interface distance). Tailor this to your own scoring metrics—e.g., predicted ΔG, known epitope coverage, or any custom structural constraints.
