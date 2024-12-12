import os, sys
import random
import yaml
import pickle
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Optional
import time
from typing import Dict, List, Tuple
import json
import math
import logging
import csv
import shutil

from google.cloud import aiplatform
from google.cloud import storage

from google.cloud.aiplatform.compat.types import (
    pipeline_job as gca_pipeline_job,
    pipeline_state as gca_pipeline_state,
)

import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

from .mutate import mutate_sequence
from .crossover import crossover_sequences
from .utils import get_sequence_similarity
from .fitness import fitness_function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure output goes to stdout
    ]
)
logger = logging.getLogger(__name__)

class JANUS:
    """ JANUS class for genetic algorithm applied on amino acid sequences.
    See example/example.py for descriptions of parameters
    """

    def __init__(
        self,
        work_dir: str,
        start_population: str,
        target_sequence: str,
        project_id: str,
        zone: str,
        pipeline_root_path: str,
        pipeline_template_path: str,
        pipeline_name: str,
        bucket_name: str,
        experiment_id: str,
        verbose_out: Optional[bool] = False,
        custom_filter: Optional[Callable] = None,
        alphabet: Optional[List[str]] = None,
        use_gpu: Optional[bool] = True,
        num_workers: Optional[int] = None,
        generations: Optional[int] = 300,
        generation_size: Optional[int] = 20,
        num_exchanges: Optional[int] = 5,
        explr_num_random_samples: Optional[int] = 5,
        explr_num_mutations: Optional[int] = 5,
        crossover_num_random_samples: Optional[int] = 1,
        exploit_num_random_samples: Optional[int] = 20,
        exploit_num_mutations: Optional[int] = 20,
        top_seqs: Optional[int] = 1,
        use_small_bfd: str = "true",
        skip_msa: str = "true",
        num_multimer_predictions_per_model: int = 1,
        is_run_relax: str = "",
        max_template_date = '2030-01-01',
        model_names = ['model_5_multimer_v3'],
        receptor_if_residues: str = ''
    ):
        if alphabet is None:
            self.alphabet = [
                'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
            ]
        else:
            self.alphabet = alphabet

        self.project_id = project_id
        self.zone = zone
        self.region = '-'.join(self.zone.split(sep='-')[:-1]) 
        self.pipeline_name = pipeline_name
        self.pipeline_template_path = pipeline_template_path
        self.pipeline_root_path = pipeline_root_path
        self.use_small_bfd = use_small_bfd
        self.skip_msa = skip_msa
        self.num_multimer_predictions_per_model = num_multimer_predictions_per_model
        self.is_run_relax = is_run_relax
        self.bucket_name = bucket_name
        self.experiment_id = experiment_id
        self.max_template_date = max_template_date
        self.model_names = model_names

        self.work_dir = work_dir
        self.fitness_function = fitness_function
        self.start_population = start_population
        self.verbose_out = verbose_out
        self.custom_filter = custom_filter
        self.use_gpu = use_gpu
        # Although we still have num_workers, we won't use multiprocessing
        self.num_workers = num_workers if num_workers is not None else 1
        self.generations = generations
        self.generation_size = generation_size
        self.num_exchanges = num_exchanges
        self.explr_num_random_samples = explr_num_random_samples
        self.explr_num_mutations = explr_num_mutations
        self.crossover_num_random_samples = crossover_num_random_samples
        self.exploit_num_random_samples = exploit_num_random_samples
        self.exploit_num_mutations = exploit_num_mutations
        self.top_seqs = top_seqs

        self.peptide_counter = 1
        self.target_sequence = target_sequence
        self.receptor_if_residues = receptor_if_residues

        os.environ['PARALLELISM'] = '25'

        if not os.path.isdir(f"{self.work_dir}"):
            os.mkdir(f"{self.work_dir}")
        self.save_hyperparameters()

        init_sequences, init_fitness = [], []
        with open(self.start_population, "r") as f:
            for line in f:
                sequence = line.strip()
                if all(aa in self.alphabet for aa in sequence):
                    init_sequences.append(sequence)
        
        logger.info(f"Initial valid sequences: {init_sequences}")
        assert (
            len(init_sequences) >= self.generation_size
        ), "Initial population smaller than generation size."
        assert (
            self.top_seqs <= self.generation_size
        ), "Number of top sequences larger than generation size."

        init_fitness = self.compute_fitness_batch(init_sequences)

        idx = np.argsort(init_fitness)[::-1]
        init_sequences = np.array(init_sequences)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_sequences[: self.generation_size]
        self.fitness = init_fitness[: self.generation_size]

        with open(os.path.join(self.work_dir, "init_seqs.txt"), "w") as f:
            f.writelines([f"{x}\n" for x in self.population])

        self.sequence_collector = {}
        uniq_pop, idx, counts = np.unique(
            self.population, return_index=True, return_counts=True
        )
        for seq, count, i in zip(uniq_pop, counts, idx):
            self._update_sequence_collector(seq, self.fitness[i], count=count)

    def _safe_fitness_value(self, fitness_val: float) -> float:
        try:
            if fitness_val is None:
                return float('-inf')
            fitness_val = float(fitness_val)
            if math.isnan(fitness_val) or math.isinf(fitness_val):
                return float('-inf')
            if fitness_val > 1e308 or fitness_val < -1e308:
                return float('-inf')
            return fitness_val
        except Exception as e:
            logger.error(f"Error in _safe_fitness_value: {e}", exc_info=True)
            return float('-inf')

    def compute_fitness_batch(self, sequences: List[str]) -> List[float]:
        if not sequences:
            logger.warning("Empty sequence list provided to compute_fitness_batch")
            return []
        
        logger.info(f"Starting batch fitness computation for {len(sequences)} sequences")
        try:
            # Validate sequences before processing
            valid_sequences = []
            for seq in sequences:
                if not isinstance(seq, str):
                    logger.error(f"Invalid sequence type: {type(seq)}")
                    continue
                if not all(aa in self.alphabet for aa in seq):
                    logger.error(f"Invalid amino acids in sequence: {seq}")
                    continue
                valid_sequences.append(seq)
            
            if not valid_sequences:
                logger.error("No valid sequences to process")
                return [float('-inf')] * len(sequences)
            
            try:
                aiplatform.init(
                    project=self.project_id,
                    location=self.region,
                    staging_bucket=f'gs://{self.bucket_name}/staging'
                )
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
                return [float('-inf')] * len(sequences)

            try:
                storage_client = storage.Client(project=self.project_id)
                bucket = storage_client.bucket(self.bucket_name)
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)
                return [float('-inf')] * len(sequences)

            sequence_paths = {}
            pipeline_jobs: Dict[str, Tuple[aiplatform.PipelineJob, str, str]] = {}
            
            for seq in valid_sequences:
                try:
                    seq_id = f'peptide_{self.peptide_counter}'
                    gcs_path = f'sequences/{self.experiment_id}/{seq_id}.fasta'

                    try:
                        fasta_content = ""
                        for i, chain_seq in enumerate(self.target_sequence):
                            if not isinstance(chain_seq, str) or not chain_seq:
                                raise ValueError(f"Invalid target sequence at index {i}")
                            chain_label = chr(65 + i)
                            fasta_content += f'>{chain_label} target_chain_{chain_label}\n{chain_seq}\n'

                        peptide_chain_label = chr(65 + len(self.target_sequence))
                        fasta_content += f'>{peptide_chain_label} {seq_id}\n{seq}\n'
                    except Exception as e:
                        logger.error(f"Error constructing FASTA for {seq}: {e}", exc_info=True)
                        continue

                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            blob = bucket.blob(gcs_path)
                            blob.upload_from_string(fasta_content)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to upload sequence {seq} after {max_retries} attempts: {e}", exc_info=True)
                                continue
                            time.sleep(2 ** attempt)

                    sequence_paths[seq] = f'gs://{self.bucket_name}/{gcs_path}'
                    self.peptide_counter += 1

                    timestamp = int(time.time() * 1000)
                    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
                    job_id = f'alphafold-multimer-{timestamp}-{random_suffix}'

                    labels = {'experiment_id': f'{self.experiment_id}_{self.peptide_counter}', 'sequence_id': f'{seq_id}'}
                    job = aiplatform.PipelineJob(
                        display_name=self.pipeline_name,
                        template_path=self.pipeline_template_path,
                        pipeline_root=self.pipeline_root_path,
                        job_id=job_id,
                        parameter_values={
                            "sequence_path": sequence_paths[seq],
                            "max_template_date": self.max_template_date,
                            "project": self.project_id,
                            "region": self.region,
                            "use_small_bfd": self.use_small_bfd,
                            "skip_msa": self.skip_msa,
                            'num_multimer_predictions_per_model': self.num_multimer_predictions_per_model,
                            'is_run_relax': self.is_run_relax,
                            'model_names': self.model_names
                        },
                        enable_caching=True,
                        labels=labels
                    )
                    
                    job.submit()
                    pipeline_run_name = job.name
                    pipeline_jobs[seq] = (job, job.resource_name, pipeline_run_name)
                    logger.info(f"Launched pipeline for sequence {seq} with run name: {pipeline_run_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing sequence {seq}: {e}", exc_info=True)
                    continue

            results: Dict[str, float] = {}
            pending_jobs = set(valid_sequences)
            
            logger.info(f"All {len(pending_jobs)} pipeline jobs submitted. Beginning monitoring phase...")
            last_status_log = time.time()
            status_log_interval = 300  # Log complete status every 5 minutes

            while pending_jobs:
                completed_seqs = set()
                current_time = time.time()
                
                # Detailed status log every 5 minutes
                if current_time - last_status_log >= status_log_interval:
                    logger.info(f"Status update - {len(pending_jobs)} jobs remaining:")
                    status_counts = {}
                    for seq in pending_jobs:
                        try:
                            if seq in pipeline_jobs:
                                status = pipeline_jobs[seq][0].state
                                status_counts[status] = status_counts.get(status, 0) + 1
                            else:
                                status_counts['NO_JOB'] = status_counts.get('NO_JOB', 0) + 1
                        except Exception as e:
                            status_counts['ERROR'] = status_counts.get('ERROR', 0) + 1
                            logger.error(f"Error getting status for {seq}: {e}")
                    
                    for status, count in status_counts.items():
                        logger.info(f"  {status}: {count} jobs")
                    last_status_log = current_time
                
                for seq in pending_jobs:
                    try:
                        if seq not in pipeline_jobs:
                            logger.warning(f"No pipeline job found for sequence {seq}")
                            self._update_sequence_collector(seq, float('-inf'))
                            completed_seqs.add(seq)
                            continue

                        job, resource_name, pipeline_run_name = pipeline_jobs[seq]
                        
                        try:
                            status = job.state
                        except Exception as e:
                            logger.error(f"Error getting job state for sequence {seq}: {e}", exc_info=True)
                            self._update_sequence_collector(seq, float('-inf'))
                            completed_seqs.add(seq)
                            continue

                        if status == gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED:
                            logger.info(f"Pipeline succeeded for sequence {seq}")
                            try:
                                result = self._process_predictions(job, seq, pipeline_jobs, pipeline_run_name=pipeline_jobs[seq][2])
                                logger.info(f"Sequence {seq} completed with fitness: {result}")
                                results[seq] = result
                                completed_seqs.add(seq)
                            except Exception as e:
                                logger.error(f"Error processing successful pipeline for sequence {seq}: {e}", exc_info=True)
                                self._update_sequence_collector(seq, float('-inf'))
                                completed_seqs.add(seq)
                        
                        elif status in [
                            gca_pipeline_state.PipelineState.PIPELINE_STATE_FAILED,
                            gca_pipeline_state.PipelineState.PIPELINE_STATE_CANCELLED
                        ]:
                            logger.error(f"Pipeline {status} for sequence {seq}")
                            self._update_sequence_collector(seq, float('-inf'))
                            completed_seqs.add(seq)
                        
                    except Exception as e:
                        logger.critical(f"Critical error processing sequence {seq}: {e}", exc_info=True)
                        self._update_sequence_collector(seq, float('-inf'))
                        completed_seqs.add(seq)

                try:
                    pending_jobs -= completed_seqs
                except Exception as e:
                    logger.error(f"Error updating pending jobs: {e}", exc_info=True)
                    pending_jobs = {seq for seq in pending_jobs if seq not in completed_seqs}
                    
                if pending_jobs:
                    logger.info(f"Waiting for {len(pending_jobs)} jobs to complete...")
                    time.sleep(60)

            logger.info("All pipeline jobs completed.")
            return [results.get(seq, float('-inf')) for seq in valid_sequences]

        except Exception as e:
            logger.critical(f"Critical error in compute_fitness_batch: {e}", exc_info=True)
            return [float('-inf')] * len(sequences)

    def _download_and_parse_result(self, output_uri: str):
        if not output_uri or not isinstance(output_uri, str):
            raise ValueError("Invalid output URI")
        logger.info(f"Downloading and parsing result from {output_uri}")
        try:
            if not output_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI format: {output_uri}")

            parts = output_uri.split('/')
            if len(parts) < 4:
                raise ValueError(f"Invalid GCS path format: {output_uri}")

            bucket_name = parts[2]
            blob_path = '/'.join(parts[3:])

            max_retries = 3
            content = None
            for attempt in range(max_retries):
                try:
                    storage_client = storage.Client(project=self.project_id)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    
                    if not blob.exists():
                        raise FileNotFoundError(f"Blob does not exist: {output_uri}")
                    
                    content = blob.download_as_bytes()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
                    logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)

            if blob_path.endswith('.json'):
                try:
                    result = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON content: {str(e)}")
                except UnicodeDecodeError as e:
                    raise ValueError(f"Failed to decode JSON content: {str(e)}")
            elif blob_path.endswith('.pkl'):
                try:
                    result = pickle.loads(content)
                except (pickle.UnpicklingError, ModuleNotFoundError) as e:
                    raise ValueError(f"Invalid pickle content or missing module: {str(e)}")
                except Exception as e:
                    raise ValueError(f"Unknown error while unpickling: {str(e)}")
            else:
                raise ValueError(f"Unsupported file format: {blob_path}")

            return result

        except Exception as e:
            logger.error(f"Error in _download_and_parse_result: {str(e)}", exc_info=True)
            raise

    def mutate_seq_list(self, seq_list: List[str], space="local"):
        logger.info(f"Starting mutation of sequences in {space} space without multiprocessing.")
        if space == "local":
            num_random_samples = self.exploit_num_random_samples
            num_mutations = self.exploit_num_mutations
        elif space == "explore":
            num_random_samples = self.explr_num_random_samples
            num_mutations = self.explr_num_mutations
        else:
            raise ValueError('Invalid space, choose "local" or "explore".')

        seq_list = seq_list * num_random_samples

        # Instead of multiprocessing, do it in a loop
        mutated_sequences = []
        try:
            for s in seq_list:
                mutated = mutate_sequence(
                    s,
                    num_mutations=num_mutations,
                    valid_amino_acids=self.alphabet
                )
                mutated_sequences.extend(mutated)
        except Exception as e:
            logger.error(f"Error during mutation: {e}", exc_info=True)
            mutated_sequences = []

        logger.info("Mutation completed.")
        return mutated_sequences

    def crossover_seq_list(self, seq_list: List[str]):
        logger.info("Starting crossover of sequences without multiprocessing.")
        crossed_sequences = []
        try:
            for s in seq_list:
                crossed = crossover_sequences(
                    s,
                    crossover_num_random_samples=self.crossover_num_random_samples,
                )
                crossed_sequences.extend(crossed)
        except Exception as e:
            logger.error(f"Error during crossover: {e}", exc_info=True)
            crossed_sequences = []

        logger.info("Crossover completed.")
        return crossed_sequences

    def check_filters(self, seq_list: List[str]):
        if self.custom_filter is not None:
            try:
                seq_list = [seq for seq in seq_list if self.custom_filter(seq)]
            except Exception as e:
                logger.error(f"Error in custom_filter: {e}. Returning unfiltered list.", exc_info=True)
        return seq_list

    def save_hyperparameters(self):
        hparams = {
            k: v if not callable(v) else v.__name__ for k, v in vars(self).items()
        }
        with open(os.path.join(self.work_dir, "hparams.yml"), "w") as f:
            yaml.dump(hparams, f)

    def run(self):
        try:
            os.makedirs(self.work_dir, exist_ok=True)
            global_csv_path = os.path.join(self.work_dir, "global_peptides.csv")
            header = ["Sequence", "Fitness", "pLDDT", "Interface Distance", "Pipeline Run Name"]
            
            # Initialize global CSV with initial population data
            initial_rows = []
            for seq in self.population:
                if seq in self.sequence_collector:
                    values = self.sequence_collector[seq]
                    # Ensure we have all required values
                    if len(values) == 5:  # [fitness, count, plddt, interface_distance, pipeline_run_name]
                        initial_rows.append([
                            seq,
                            values[0],  # fitness
                            values[2],  # plddt
                            values[3],  # interface_distance
                            values[4]   # pipeline_run_name
                        ])
            
            if initial_rows:
                logger.info(f"Initializing global CSV with {len(initial_rows)} sequences")
                self._write_csv_safely(global_csv_path, initial_rows, header)
            else:
                logger.warning("No valid initial sequences to write to global CSV")
                # Create empty CSV with header
                self._write_csv_safely(global_csv_path, [], header)
            
            for gen_ in range(self.generations):
                generation_start_time = time.time()
                
                try:
                    # Save checkpoint at start of generation
                    checkpoint_file = os.path.join(self.work_dir, "checkpoint.pkl")
                    self._save_checkpoint(checkpoint_file, gen_)
                    
                    logger.info(f"On generation {gen_}/{self.generations}")
                    
                    def safe_write_to_file(filepath: str, content: str):
                        try:
                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                            with open(filepath, "w") as f:
                                f.write(content)
                        except Exception as e:
                            logger.error(f"Error writing to {filepath}: {e}", exc_info=True)

                    if np.all(np.isinf(self.fitness) & (np.array(self.fitness) < 0)):
                        logger.warning(f"Generation {gen_}: All sequences have -inf fitness. Attempting recovery...")
                        with open(os.path.join(self.work_dir, "error_log.txt"), "a+") as f:
                            f.write(f"Generation {gen_}: All sequences had -inf fitness. Attempting recovery.\n")
                        continue

                    keep_seqs, replace_seqs = self.get_good_bad_sequences(
                        self.fitness, self.population, self.generation_size
                    )
                    replace_seqs = list(set(replace_seqs))

                    logger.info("Starting exploration phase.")
                    explr_seqs = []
                    timeout_counter = 0
                    while len(explr_seqs) < self.generation_size-len(keep_seqs):
                        mut_seq_explr = self.mutate_seq_list(
                            replace_seqs[0 : len(replace_seqs) // 2], space="explore"
                        )
                        mut_seq_explr = self.check_filters(mut_seq_explr)

                        seq_join = []
                        for item in replace_seqs[len(replace_seqs) // 2 :]:
                            seq_join.append(item + "xxx" + random.choice(keep_seqs))
                        cross_seq_explr = self.crossover_seq_list(seq_join)
                        cross_seq_explr = self.check_filters(cross_seq_explr)

                        all_seqs = list(set(mut_seq_explr + cross_seq_explr))
                        for x in all_seqs:
                            if x not in self.sequence_collector:
                                explr_seqs.append(x)
                        explr_seqs = list(set(explr_seqs))

                        timeout_counter += 1
                        if timeout_counter % 100 == 0:
                            logger.info('Exploration: {} iterations of filtering. \
                            Filter may be too strict, or you need more mutations/crossovers.'.format(timeout_counter))

                    replaced_pop = random.sample(
                        explr_seqs, self.generation_size - len(keep_seqs)
                    )

                    self.population = keep_seqs + replaced_pop

                    sequences_to_evaluate = []
                    self.fitness = []
                    for seq in self.population:
                        if seq not in self.sequence_collector:
                            sequences_to_evaluate.append(seq)
                        else:
                            self._update_sequence_collector(seq, self.sequence_collector[seq][0], count=self.sequence_collector[seq][1] + 1)
                    
                    if sequences_to_evaluate:
                        batch_fitness = self.compute_fitness_batch(sequences_to_evaluate)
                        for seq, fit_val in zip(sequences_to_evaluate, batch_fitness):
                            self._update_sequence_collector(seq, fit_val)
                            
                    self.fitness.extend([self.sequence_collector[seq][0] for seq in self.population])
                    
                    idx_sort = np.argsort(self.fitness)[::-1]
                    logger.info(f"(Explr) Top Fitness: {self.fitness[idx_sort[0]]}")
                    logger.info(f"(Explr) Top Sequence: {self.population[idx_sort[0]]}")

                    fitness_sort = np.array(self.fitness)[idx_sort]
                    if self.verbose_out:
                        with open(
                            os.path.join(
                                self.work_dir, str(gen_) + "_DATA", "fitness_explore.txt"
                            ),
                            "w",
                        ) as f:
                            f.writelines(["{} ".format(x) for x in fitness_sort])
                            f.writelines(["\n"])
                    else:
                        with open(os.path.join(self.work_dir, "fitness_explore.txt"), "w") as f:
                            f.writelines(["{} ".format(x) for x in fitness_sort])
                            f.writelines(["\n"])

                    population_sort = np.array(self.population)[idx_sort]
                    if self.verbose_out:
                        with open(
                            os.path.join(
                                self.work_dir, str(gen_) + "_DATA", "population_explore.txt"
                            ),
                            "w",
                        ) as f:
                            f.writelines(["{} ".format(x) for x in population_sort])
                            f.writelines(["\n"])
                    else:
                        with open(
                            os.path.join(self.work_dir, "population_explore.txt"), "w"
                        ) as f:
                            f.writelines(["{} ".format(x) for x in population_sort])
                            f.writelines(["\n"])

                    logger.info("Starting exploitation phase.")
                    exploit_seqs = []
                    timeout_counter = 0
                    while len(exploit_seqs) < self.generation_size:
                        seq_local_search = population_sort[0 : self.top_seqs].tolist()
                        mut_seq_loc = self.mutate_seq_list(seq_local_search, "local")
                        mut_seq_loc = self.check_filters(mut_seq_loc)

                        for x in mut_seq_loc:
                            if x not in self.sequence_collector:
                                exploit_seqs.append(x)

                        timeout_counter += 1
                        if timeout_counter % 100 == 0:
                            logger.info('Exploitation: {} iterations of filtering. \
                            Filter may be too strict, or you need more mutations/crossovers.'.format(timeout_counter))

                    similarity_scores = get_sequence_similarity(exploit_seqs, population_sort[0])
                    sim_sort_idx = np.argsort(similarity_scores)[::-1][: self.generation_size]
                    self.population_loc = np.array(exploit_seqs)[sim_sort_idx]

                    sequences_to_evaluate = []
                    self.fitness_loc = []
                    for seq in self.population_loc:
                        if seq not in self.sequence_collector:
                            sequences_to_evaluate.append(seq)
                        else:
                            self._update_sequence_collector(seq, self.sequence_collector[seq][0], count=self.sequence_collector[seq][1] + 1)
                    
                    if sequences_to_evaluate:
                        batch_fitness = self.compute_fitness_batch(sequences_to_evaluate)
                        for seq, fit_val in zip(sequences_to_evaluate, batch_fitness):
                            self._update_sequence_collector(seq, fit_val)
                            
                    self.fitness_loc.extend([self.sequence_collector[seq][0] for seq in self.population_loc])

                    idx_sort = np.argsort(self.fitness_loc)[::-1]
                    logger.info(f"(Local) Top Fitness: {self.fitness_loc[idx_sort[0]]}")
                    logger.info(f"(Local) Top Sequence: {self.population_loc[idx_sort[0]]}")

                    fitness_sort = np.array(self.fitness_loc)[idx_sort]
                    if self.verbose_out:
                        with open(
                            os.path.join(
                                self.work_dir, str(gen_) + "_DATA", "fitness_local_search.txt"
                            ),
                            "w",
                        ) as f:
                            f.writelines(["{} ".format(x) for x in fitness_sort])
                            f.writelines(["\n"])
                    else:
                        with open(
                            os.path.join(self.work_dir, "fitness_local_search.txt"), "w"
                        ) as f:
                            f.writelines(["{} ".format(x) for x in fitness_sort])
                            f.writelines(["\n"])

                    population_sort = np.array(self.population_loc)[idx_sort]
                    if self.verbose_out:
                        with open(
                            os.path.join(
                                self.work_dir,
                                str(gen_) + "_DATA",
                                "population_local_search.txt",
                            ),
                            "w",
                        ) as f:
                            f.writelines(["{} ".format(x) for x in population_sort])
                            f.writelines(["\n"])
                    else:
                        with open(
                            os.path.join(self.work_dir, "population_local_search.txt"), "w"
                        ) as f:
                            f.writelines(["{} ".format(x) for x in population_sort])
                            f.writelines(["\n"])

                    best_seq_local = population_sort[0 : self.num_exchanges]
                    best_fitness_local = fitness_sort[0 : self.num_exchanges]

                    idx_sort = np.argsort(self.fitness)[::-1]
                    worst_indices = idx_sort[-self.num_exchanges :]
                    for i, idx_ in enumerate(worst_indices):
                        try:
                            self.population[idx_] = best_seq_local[i]
                            self.fitness[idx_] = best_fitness_local[i]
                        except Exception as e:
                            logger.error(f"Error during population exchange: {e}", exc_info=True)
                            continue

                    fit_all_best = np.argmax(self.fitness)
                    generation_best_path = os.path.join(self.work_dir, "generation_all_best.txt")
                    safe_write_to_file(
                        generation_best_path,
                        f"Gen:{gen_}, {self.population[fit_all_best]}, {self.fitness[fit_all_best]} \n"
                    )

                    logger.info(f"Generation {gen_} summary:")
                    logger.info(f"Best fitness: {self.fitness[fit_all_best]}")
                    logger.info(f"Best sequence: {self.population[fit_all_best]}")
                    logger.info(f"Average fitness: {np.mean(self.fitness)}")
                    logger.info(f"Population size: {len(self.population)}")

                    # Update CSVs with new data
                    generation_csv_path = os.path.join(self.work_dir, f"generation_{gen_}_peptides.csv")
                    generation_rows = []
                    
                    for seq in self.population:
                        if seq in self.sequence_collector:
                            values = self.sequence_collector[seq]
                            if len(values) == 5:
                                generation_rows.append([
                                    seq,
                                    values[0],  # fitness
                                    values[2],  # plddt
                                    values[3],  # interface_distance
                                    values[4]   # pipeline_run_name
                                ])
                    
                    if generation_rows:
                        logger.info(f"Writing {len(generation_rows)} sequences to generation CSV")
                        self._write_csv_safely(generation_csv_path, generation_rows, header)
                        
                        # Update global CSV
                        try:
                            existing_rows = []
                            if os.path.exists(global_csv_path):
                                with open(global_csv_path, mode='r', newline='') as file:
                                    reader = csv.reader(file)
                                    next(reader)  # Skip header
                                    existing_rows = list(reader)
                            
                            # Combine and sort all rows
                            all_rows = existing_rows + generation_rows
                            # Sort by fitness, handling potential invalid values
                            sorted_rows = sorted(
                                all_rows,
                                key=lambda x: float(x[1]) if x[1] and x[1] != '' else float('-inf'),
                                reverse=True
                            )
                            
                            # Remove duplicates based on sequence
                            seen_sequences = set()
                            unique_rows = []
                            for row in sorted_rows:
                                if row[0] not in seen_sequences:
                                    seen_sequences.add(row[0])
                                    unique_rows.append(row)
                            
                            logger.info(f"Writing {len(unique_rows)} unique sequences to global CSV")
                            self._write_csv_safely(global_csv_path, unique_rows, header)
                            
                        except Exception as e:
                            logger.error(f"Error updating global CSV: {e}", exc_info=True)
                    else:
                        logger.warning(f"No valid sequences to write for generation {gen_}")
                    
                except Exception as e:
                    logger.error(f"Error in generation {gen_}: {e}", exc_info=True)
                    if os.path.exists(checkpoint_file):
                        self._load_checkpoint(checkpoint_file)
                    continue
                    
                finally:
                    generation_time = time.time() - generation_start_time
                    logger.info(f"Generation {gen_} completed in {generation_time:.2f} seconds")

        except Exception as e:
            logger.critical(f"Critical error in run(): {e}", exc_info=True)
            raise

        logger.info("Run completed successfully.")
        return

    @staticmethod
    def get_good_bad_sequences(fitness, population, generation_size):
        fitness = np.array(fitness)
        
        if np.all(np.isinf(fitness) & (fitness < 0)):
            print("Warning: All sequences have -inf fitness. Generating new random sequences...")
            new_sequences = []
            sequence_length = len(population[0]) if population else 20
            try:
                for _ in range(generation_size):
                    new_seq = ''.join(random.choices(JANUS.alphabet, k=sequence_length))
                    new_sequences.append(new_seq)
                return new_sequences[:math.ceil(generation_size * 0.2)], new_sequences
            except Exception as e:
                print(f"Error generating new sequences: {e}")
                keep_idx = math.ceil(len(population) * 0.2)
                return population[:keep_idx], population[keep_idx:]

        idx_sort = fitness.argsort()[::-1]
        keep_ratio = 0.2
        keep_idx = math.ceil(len(list(idx_sort)) * keep_ratio)

        try:
            F_50_val = fitness[idx_sort[keep_idx]]
            F_25_val = np.array(fitness) - F_50_val
            F_25_val = np.array([x for x in F_25_val if x < 0]) + F_50_val
            F_25_sort = F_25_val.argsort()[::-1]
            F_25_val = F_25_val[F_25_sort[0]]

            denominator = F_50_val - F_25_val
            if denominator == 0:
                prob_ = np.ones(len(fitness)) / len(fitness)
            else:
                prob_ = 1.0 / (3.0 ** ((F_50_val - fitness) / denominator) + 1)
            
            prob_sum = sum(prob_)
            if prob_sum == 0:
                prob_ = np.ones(len(prob_)) / len(prob_)
            else:
                prob_ = prob_ / prob_sum

            to_keep = np.random.choice(generation_size, keep_idx, p=prob_)
            to_replace = [i for i in range(generation_size) if i not in to_keep][
                0 : generation_size - len(to_keep)
            ]

            keep_seqs = [population[i] for i in to_keep]
            replace_seqs = [population[i] for i in to_replace]

            best_seq = population[idx_sort[0]]
            if best_seq not in keep_seqs:
                keep_seqs.append(best_seq)
                if best_seq in replace_seqs:
                    replace_seqs.remove(best_seq)

            if not keep_seqs or not replace_seqs:
                raise Exception("Badly sampled population!")
            
        except Exception as e:
            print(f"Error in probability calculation: {e}. Falling back to simple sorting.")
            keep_seqs = [population[i] for i in idx_sort[:keep_idx]]
            replace_seqs = [population[i] for i in idx_sort[keep_idx:]]

        return keep_seqs, replace_seqs

    def log(self):
        pass

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]

    def _process_predictions(self, job, seq, pipeline_jobs, pipeline_run_name):
        """Process predictions with proper error handling and sequence_collector updates."""
        if not hasattr(self, 'sequence_collector'):
            logger.error("sequence_collector not initialized")
            self.sequence_collector = {}
        
        if not hasattr(job, 'state'):
            logger.error("job state not initialized")
            return float('-inf')
        
        if job.state != gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED:
            logger.error(f"Pipeline state is not succeeded for {seq}")
            return float('-inf')
        
        logger.info(f"Processing predictions for {seq}")
        task_details = job.task_details
        features_dir = ""
            
        for task in task_details:
            if task.task_name == 'create-run-id':
                try:
                    metadata_dict = dict(task.execution.metadata)
                    output_str = metadata_dict.get('output:Output', '{}')
                    output_json = json.loads(output_str)
                    features_dir = output_json.get('full_protein', '')
                except Exception as e:
                    logger.error(f"Error parsing create-run-id output for {seq}: {e}", exc_info=True)
                    return float('-inf')

        if not features_dir:
            logger.error(f"No features directory found for {seq}")
            return float('-inf')

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        prefix = f"pipeline_runs/{self.pipeline_name}/16853584617/{pipeline_jobs[seq][2]}/"
            
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            logger.error(f"No blobs found for {seq}")
            return float('-inf')

        predict_dirs = list(set(
            "/".join(blob.name.split('/')[:-1]) 
            for blob in blobs
            if 'predict_' in blob.name and 'executor_output.json' in blob.name
        ))

        if not predict_dirs:
            logger.error(f"No prediction directories found for {seq}")
            return float('-inf')
        
        try:
            # Add validation for predict_dirs format
            if not isinstance(predict_dirs, (list, tuple)):
                logger.error(f"Invalid predict_dirs type: {type(predict_dirs)}")
                return float('-inf')
            
            max_confidence = float('-inf')
            best_prediction_path = None
            
            if not features_dir.startswith('gs://'):
                features_dir = f"gs://{self.bucket_name}/{features_dir}"
            
            for predict_dir in predict_dirs:
                try:
                    executor_output_path = f"{predict_dir}/executor_output.json"
                    executor_output = self._download_and_parse_result(
                        f"gs://{self.bucket_name}/{executor_output_path}"
                    )
                    confidence = executor_output['artifacts']['raw_prediction']['artifacts'][0]['metadata'].get(
                        'ranking_confidence', None
                    )
                    if confidence is not None and confidence > max_confidence:
                        max_confidence = confidence
                        best_prediction_path = f"gs://{self.bucket_name}/{predict_dir}/raw_prediction.pkl"
                except Exception as e:
                    logger.error(f"Error processing prediction directory {predict_dir}: {e}", exc_info=True)
                    continue

            if best_prediction_path is None:
                return float('-inf')

            try:
                feature_dict = self._download_and_parse_result(f"{features_dir}/all_chain_features.pkl")
                prediction_result = self._download_and_parse_result(best_prediction_path)

                if not isinstance(feature_dict, dict) or not isinstance(prediction_result, dict):
                    logger.error(f"Invalid feature_dict or prediction_result type for {seq}")
                    return float('-inf')

                try:
                    logger.info(f"Computing fitness for sequence {seq}")
                    fitness_val, interface_distance, plddt = self.fitness_function(
                        seq, 
                        self.receptor_if_residues, 
                        feature_dict, 
                        prediction_result
                    )
                    # Clear large objects from memory immediately after use
                    del feature_dict
                    del prediction_result
                    
                    logger.info(f"Fitness calculation results for {seq}:")
                    logger.info(f"Raw fitness: {fitness_val}")
                    logger.info(f"Interface distance: {interface_distance}")
                    logger.info(f"pLDDT score: {plddt}")
                    
                    # Update sequence collector
                    self._update_sequence_collector(
                        seq, 
                        fitness_val, 
                        count=1, 
                        plddt=plddt, 
                        interface_distance=interface_distance, 
                        pipeline_run_name=pipeline_run_name
                    )
                    
                    return self._safe_fitness_value(fitness_val)

                except Exception as e:
                    logger.error(f"Error in fitness calculation for {seq}: {e}", exc_info=True)
                    self._update_sequence_collector(seq, float('-inf'))
                    return float('-inf')

            except Exception as e:
                logger.error(f"Error in _process_predictions for {seq}: {e}", exc_info=True)
                return float('-inf')

        except Exception as e:
            logger.error(f"Error in _process_predictions for {seq}: {e}", exc_info=True)
            return float('-inf')

    def _save_checkpoint(self, checkpoint_file: str, generation: int):
        """Save current state to checkpoint file."""
        try:
            checkpoint_data = {
                'generation': generation,
                'population': self.population,
                'fitness': self.fitness,
                'sequence_collector': self.sequence_collector,
                'peptide_counter': self.peptide_counter
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved for generation {generation}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}", exc_info=True)

    def _load_checkpoint(self, checkpoint_file: str):
        """Load state from checkpoint file."""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            self.population = checkpoint_data['population']
            self.fitness = checkpoint_data['fitness']
            self.sequence_collector = checkpoint_data['sequence_collector']
            self.peptide_counter = checkpoint_data['peptide_counter']
            logger.info(f"Checkpoint loaded from generation {checkpoint_data['generation']}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)

    def _update_sequence_collector(self, seq: str, fitness_val: float, count: int = 1, 
                                 plddt: float = None, interface_distance: float = None, 
                                 pipeline_run_name: str = None):
        """Safely update sequence collector with validation."""
        try:
            if not isinstance(seq, str):
                raise ValueError(f"Invalid sequence type: {type(seq)}")
            
            fitness_val = self._safe_fitness_value(fitness_val)
            
            if seq not in self.sequence_collector:
                self.sequence_collector[seq] = [fitness_val, count, plddt, interface_distance, pipeline_run_name]
            else:
                current_values = self.sequence_collector[seq]
                new_count = current_values[1] + count
                # Only update other values if they're not None
                new_plddt = plddt if plddt is not None else current_values[2]
                new_interface_distance = interface_distance if interface_distance is not None else current_values[3]
                new_pipeline_run_name = pipeline_run_name if pipeline_run_name is not None else current_values[4]
                
                self.sequence_collector[seq] = [fitness_val, new_count, new_plddt, 
                                              new_interface_distance, new_pipeline_run_name]
                
            logger.debug(f"Updated sequence_collector for {seq}: {self.sequence_collector[seq]}")
            
        except Exception as e:
            logger.error(f"Error updating sequence_collector for {seq}: {e}", exc_info=True)
            # Ensure we at least have a valid entry
            self.sequence_collector[seq] = [float('-inf'), 1, None, None, None]

    def _write_csv_safely(self, filepath: str, rows: List[List], header: List[str]):
        """Safely write data to CSV with validation."""
        try:
            if not rows:
                logger.warning(f"No rows to write to {filepath}")
                return
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create backup of existing file if it exists
            if os.path.exists(filepath):
                backup_path = f"{filepath}.backup"
                shutil.copy2(filepath, backup_path)
            
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                
                for row in rows:
                    # Validate row length matches header
                    if len(row) != len(header):
                        logger.warning(f"Row length mismatch: {row}")
                        row = row + [None] * (len(header) - len(row))  # Pad with None
                    writer.writerow(row)
                
            logger.info(f"Successfully wrote {len(rows)} rows to {filepath}")
            
        except Exception as e:
            logger.error(f"Error writing to CSV {filepath}: {e}", exc_info=True)
            # Restore from backup if available
            backup_path = f"{filepath}.backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, filepath)
                logger.info(f"Restored {filepath} from backup")
