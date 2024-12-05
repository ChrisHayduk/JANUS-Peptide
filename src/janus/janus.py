import os, sys
import multiprocessing
import random
import yaml
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Optional
from google.cloud import aiplatform
from google.cloud import storage
import time
from typing import Dict, List, Tuple
import concurrent.futures
import json
import math

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
        num_multimer_predictions_per_model: int = 1,
        is_run_relax: str = "",
        max_template_date = '2030-01-01',
        model_names = ['model_5_multimer_v3'],
        receptor_if_residues: str = ''
    ):
        # Default amino acid alphabet if none provided
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
        self.num_multimer_predictions_per_model = num_multimer_predictions_per_model
        self.is_run_relax = is_run_relax
        self.bucket_name = bucket_name
        self.experiment_id = experiment_id
        self.max_template_date = max_template_date
        self.model_names = model_names

        # set all class variables
        self.work_dir = work_dir
        self.fitness_function = fitness_function
        self.start_population = start_population
        self.verbose_out = verbose_out
        self.custom_filter = custom_filter
        self.use_gpu = use_gpu
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
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

        os.environ['PARALLELISM'] = '20'

        # create dump folder
        if not os.path.isdir(f"{self.work_dir}"):
            os.mkdir(f"{self.work_dir}")
        self.save_hyperparameters()

        # get initial population
        init_sequences, init_fitness = [], []
        with open(self.start_population, "r") as f:
            for line in f:
                sequence = line.strip()
                if all(aa in self.alphabet for aa in sequence):
                    init_sequences.append(sequence)
        
        print(init_sequences)
        # check that parameters are valid
        assert (
            len(init_sequences) >= self.generation_size
        ), "Initial population smaller than generation size."
        assert (
            self.top_seqs <= self.generation_size
        ), "Number of top sequences larger than generation size."

        # get initial fitness
        init_fitness = self.compute_fitness_batch(init_sequences)

        # sort the initial population and save in class
        idx = np.argsort(init_fitness)[::-1]
        init_sequences = np.array(init_sequences)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_sequences[: self.generation_size]
        self.fitness = init_fitness[: self.generation_size]

        with open(os.path.join(self.work_dir, "init_seqs.txt"), "w") as f:
            f.writelines([f"{x}\n" for x in self.population])

        # store in collector, deal with duplicates
        self.sequence_collector = {}
        uniq_pop, idx, counts = np.unique(
            self.population, return_index=True, return_counts=True
        )
        for seq, count, i in zip(uniq_pop, counts, idx):
            self.sequence_collector[seq] = [self.fitness[i], count]

    def compute_fitness_batch(self, sequences: List[str]) -> List[float]:
        """Compute fitness for a batch of sequences using Vertex AI pipeline.
        
        Args:
            sequences (List[str]): List of sequences to evaluate
            
        Returns:
            List[float]: Fitness values for each sequence
        """
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f'gs://{self.bucket_name}/staging'
        )

        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)

        # Launch pipeline jobs in parallel
        pipeline_jobs: Dict[str, Tuple[aiplatform.PipelineJob, str]] = {}
        
        print(f"Launching {len(sequences)} Vertex AI pipeline jobs...")
        # Upload sequences to GCS and track their paths
        sequence_paths = {}
        for seq in sequences:
            # Create unique path for each sequence
            seq_id = f'peptide_{self.peptide_counter}'
            gcs_path = f'sequences/{self.experiment_id}/{seq_id}.fasta'
            
            # Create multi-chain FASTA content
            # Chain A is the target protein, Chain B is the peptide
            fasta_content = f'>A target_protein\n{self.target_sequence}\n>B {seq_id}\n{seq}\n'
            
            # Upload to GCS
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(fasta_content)
            
            # Store full GCS path
            sequence_paths[seq] = f'gs://{self.bucket_name}/{gcs_path}'
            self.peptide_counter += 1

            # Create unique path for each sequence
            timestamp = int(time.time() * 1000)  # millisecond timestamp
            random_suffix = ''.join(random.choices('0123456789abcdef', k=6))  # random hex string

            # Create unique job ID for each pipeline
            job_id = f'alphafold-multimer-{timestamp}-{random_suffix}'

            labels = {'experiment_id': f'{self.experiment_id}_{self.peptide_counter}', 'sequence_id': f'{seq_id}'}
            # Assume pipeline parameters are configured through a template
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
                    'num_multimer_predictions_per_model': self.num_multimer_predictions_per_model,
                    'is_run_relax': self.is_run_relax,
                    'model_names': self.model_names
                },
                enable_caching=False,
                labels=labels
            )
            
            # Submit job
            job.submit()
            pipeline_run_name = job.name  # This gets the generated pipeline run name
            pipeline_jobs[seq] = (job, job.resource_name, pipeline_run_name)
            print(f"Launched pipeline for sequence {seq} with run name: {pipeline_run_name}")
            
        # Monitor jobs and collect results
        results: Dict[str, float] = {}
        pending_jobs = set(sequences)
        
        print("Waiting for pipeline jobs to complete...")
        while pending_jobs:
            completed_seqs = set()
            
            for seq in pending_jobs:
                job, resource_name, pipeline_run_name = pipeline_jobs[seq]  # Unpack all three values
                
                # Check job status
                status = job.state
                print(f"Status for sequence {seq}: {status}")
                
                if status == gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED:
                    print(f"Pipeline suceeded for sequence {seq}")

                    # Access task details to find the output of the specific task
                    task_details = job.task_details

                    features_dir = ""
                        
                    # Iterate over task details to find the task with the desired output
                    for task in task_details:
                        if task.task_name == 'create-run-id':

                            metadata_dict = dict(task.execution.metadata)

                            output_str = metadata_dict['output:Output']
                            output_json = json.loads(output_str)
                            features_dir = output_json['full_protein']
                            break


                    print(f"Features directory for sequence {seq}: {features_dir}")
                    
                    try:
                        base_uri = f'gs://{self.bucket_name}/pipeline_runs/{self.pipeline_name}/16853584617/{pipeline_jobs[seq][2]}'
                        
                        # List all predict_* directories
                        storage_client = storage.Client(project=self.project_id)
                        bucket = storage_client.bucket(self.bucket_name)
                        prefix=f"pipeline_runs/{self.pipeline_name}/16853584617/{pipeline_jobs[seq][2]}/"
                        
                        blobs = bucket.list_blobs(prefix=prefix)

                        predict_dirs = list(set(
                            "/".join(blob.name.split('/')[:-1]) for blob in blobs
                            if 'predict_' in blob.name and 'executor_output.json' in blob.name
                        ))
                        
                        # Find prediction with highest ranking confidence
                        max_confidence = float('-inf')
                        best_prediction_path = None
                        
                        for predict_dir in predict_dirs:
                            # Get the executor output JSON
                            executor_output_path = f"{predict_dir}/executor_output.json"
                            executor_output = self._download_and_parse_result(f"gs://{self.bucket_name}/{executor_output_path}")
                            
                            # Extract ranking confidence
                            try:
                                confidence = executor_output['artifacts']['raw_prediction']['artifacts'][0]['metadata']['ranking_confidence']
                                if confidence > max_confidence:
                                    max_confidence = confidence
                                    # Get path to raw_prediction.pkl in the same directory
                                    best_prediction_path = f"gs://{self.bucket_name}/{predict_dir}/raw_prediction.pkl"
                            except (KeyError, IndexError) as e:
                                print(f"Warning: Could not extract ranking confidence from {executor_output_path}: {e}")
                                continue
                        
                        if best_prediction_path is None:
                            raise ValueError("No valid predictions found")
                            
                        # Load the best prediction and features
                        prediction_result = self._download_and_parse_result(best_prediction_path)
                        feature_dict = self._download_and_parse_result(f"{features_dir}/all_chain_features.pkl")
                        
                        # Run fitness function on the AlphaFold2 output
                        fitness, if_dist_peptide, plddt, unrelaxed_protein = self.fitness_function(
                            seq, 
                            self.receptor_if_residues, 
                            feature_dict, 
                            prediction_result
                        )
                        results[seq] = fitness
                        completed_seqs.add(seq)
                        print(f"Successfully processed sequence {seq} with fitness {fitness}")
                        print(f'pLDDT: {plddt}, if_dist_peptide: {if_dist_peptide}')
                        
                    except Exception as e:
                        print(f"Error processing results for sequence {seq}: {e}")
                        # Assign worst possible fitness on failure
                        results[seq] = float('-inf')
                        completed_seqs.add(seq)
                
                elif status == gca_pipeline_state.PipelineState.PIPELINE_STATE_FAILED:
                    print(f"Pipeline failed for sequence {seq}")
                    results[seq] = float('-inf')
                    completed_seqs.add(seq)
                
                elif status == gca_pipeline_state.PipelineState.PIPELINE_STATE_CANCELLED:
                    print(f"Pipeline cancelled for sequence {seq}")
                    results[seq] = float('-inf')
                    completed_seqs.add(seq)
                
                elif status == gca_pipeline_state.PipelineState.PIPELINE_STATE_PAUSED:
                    print(f"Pipeline paused for sequence {seq}")
                    # Don't mark as completed, will check again next iteration
                
                elif status == gca_pipeline_state.PipelineState.PIPELINE_STATE_QUEUED:
                    print(f"Pipeline queued for sequence {seq}")
                    # Don't mark as completed, will check again next iteration
                
                elif status == gca_pipeline_state.PipelineState.PIPELINE_STATE_RUNNING:
                    print(f"Pipeline still running for sequence {seq}")
                    # Don't mark as completed, will check again next iteration
                
                else:
                    print(f"Unknown pipeline state {status} for sequence {seq}")
                    # Don't mark as completed, will check again next iteration

            # Remove completed jobs from pending set
            pending_jobs -= completed_seqs
            
            if pending_jobs:
                print(f"Waiting for {len(pending_jobs)} jobs to complete...")
                time.sleep(60)  # Wait before checking again
                
        # Return results in same order as input sequences
        return [results[seq] for seq in sequences]

    def _download_and_parse_result(self, output_uri: str):
        """Download and parse results from GCS.
        
        Args:
            output_uri (str): GCS URI containing pipeline outputs
                
        Returns:
            Dict containing parsed results (either prediction or features)
        """
        from google.cloud import storage
        import pickle
        import json
        
        # Parse bucket and blob path from uri
        # gs://bucket-name/path/to/file
        bucket_name = output_uri.split('/')[2]
        blob_path = '/'.join(output_uri.split('/')[3:])
        
        # Download results from GCS
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download content as bytes
        content = blob.download_as_bytes()
        
        # Parse based on file extension
        if blob_path.endswith('.json'):
            result = json.loads(content.decode('utf-8'))
        elif blob_path.endswith('.pkl'):
            result = pickle.loads(content)
        else:
            raise ValueError(f"Unsupported file format for {blob_path}. Expected .json or .pkl")
                
        return result

    def mutate_seq_list(self, seq_list: List[str], space="local"):
        # parallelized mutation function
        if space == "local":
            num_random_samples = self.exploit_num_random_samples
            num_mutations = self.exploit_num_mutations
        elif space == "explore":
            num_random_samples = self.explr_num_random_samples
            num_mutations = self.explr_num_mutations
        else:
            raise ValueError('Invalid space, choose "local" or "explore".')

        seq_list = seq_list * num_random_samples
        with multiprocessing.Pool(self.num_workers) as pool:
            mut_seq_list = pool.map(
                partial(
                    mutate_sequence,
                    num_mutations=num_mutations,
                    valid_amino_acids=self.alphabet
                ),
                seq_list,
            )
        mut_seq_list = self.flatten_list(mut_seq_list)
        return mut_seq_list

    def crossover_seq_list(self, seq_list: List[str]):
        # parallelized crossover function
        with multiprocessing.Pool(self.num_workers) as pool:
            cross_seq = pool.map(
                partial(
                    crossover_sequences,
                    crossover_num_random_samples=self.crossover_num_random_samples,
                ),
                seq_list,
            )
        cross_seq = self.flatten_list(cross_seq)
        return cross_seq

    def check_filters(self, seq_list: List[str]):
        if self.custom_filter is not None:
            seq_list = [seq for seq in seq_list if self.custom_filter(seq)]
        return seq_list

    def save_hyperparameters(self):
        hparams = {
            k: v if not callable(v) else v.__name__ for k, v in vars(self).items()
        }
        with open(os.path.join(self.work_dir, "hparams.yml"), "w") as f:
            yaml.dump(hparams, f)

    def run(self):
        """ Run optimization based on hyperparameters initialized
        """
        for gen_ in range(self.generations):
            # bookkeeping
            if self.verbose_out:
                output_dir = os.path.join(self.work_dir, f"{gen_}_DATA")
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)

            print(f"On generation {gen_}/{self.generations}")

            keep_seqs, replace_seqs = self.get_good_bad_sequences(
                self.fitness, self.population, self.generation_size
            )
            replace_seqs = list(set(replace_seqs))

            ### EXPLORATION ###
            explr_seqs = []
            timeout_counter = 0
            while len(explr_seqs) < self.generation_size-len(keep_seqs):
                # Mutations:
                mut_seq_explr = self.mutate_seq_list(
                    replace_seqs[0 : len(replace_seqs) // 2], space="explore"
                )
                mut_seq_explr = self.check_filters(mut_seq_explr)

                # Crossovers:
                seq_join = []
                for item in replace_seqs[len(replace_seqs) // 2 :]:
                    seq_join.append(item + "xxx" + random.choice(keep_seqs))
                cross_seq_explr = self.crossover_seq_list(seq_join)
                cross_seq_explr = self.check_filters(cross_seq_explr)

                # Combine and get unique sequences not yet found
                all_seqs = list(set(mut_seq_explr + cross_seq_explr))
                for x in all_seqs:
                    if x not in self.sequence_collector:
                        explr_seqs.append(x)
                explr_seqs = list(set(explr_seqs))

                timeout_counter += 1
                if timeout_counter % 100 == 0:
                    print(f'Exploration: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            replaced_pop = random.sample(
                explr_seqs, self.generation_size - len(keep_seqs)
            )


            # Calculate actual fitness for the exploration population
            self.population = keep_seqs + replaced_pop

            # Replace individual fitness calls with batched version
            sequences_to_evaluate = []
            self.fitness = []
            for seq in self.population:
                if seq not in self.sequence_collector:
                    sequences_to_evaluate.append(seq)
                else:
                    self.sequence_collector[seq][1] += 1
            
            if sequences_to_evaluate:
                # Batch evaluate fitness using Vertex AI
                batch_fitness = self.compute_fitness_batch(sequences_to_evaluate)
                
                # Update sequence collector and fitness list
                for seq, fitness in zip(sequences_to_evaluate, batch_fitness):
                    self.sequence_collector[seq] = [fitness, 1]
                    
            self.fitness.extend([self.sequence_collector[seq][0] for seq in self.population])
            
            # Print exploration data
            idx_sort = np.argsort(self.fitness)[::-1]
            print(f"    (Explr) Top Fitness: {self.fitness[idx_sort[0]]}")
            print(f"    (Explr) Top Sequence: {self.population[idx_sort[0]]}")

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

            ### EXPLOITATION ###
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
                    print(f'Exploitation: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            # sort by similarity, only keep ones similar to best
            similarity_scores = get_sequence_similarity(exploit_seqs, population_sort[0])
            sim_sort_idx = np.argsort(similarity_scores)[::-1][: self.generation_size]
            self.population_loc = np.array(exploit_seqs)[sim_sort_idx]

            # Calculate fitness for local search
            sequences_to_evaluate = []
            self.fitness_loc = []
            for seq in self.population_loc:
                if seq not in self.sequence_collector:
                    sequences_to_evaluate.append(seq)
                else:
                    self.sequence_collector[seq][1] += 1
            
            if sequences_to_evaluate:
                # Batch evaluate fitness using Vertex AI
                batch_fitness = self.compute_fitness_batch(sequences_to_evaluate)
                
                # Update sequence collector and fitness list
                for seq, fitness in zip(sequences_to_evaluate, batch_fitness):
                    self.sequence_collector[seq] = [fitness, 1]
                    
            self.fitness_loc.extend([self.sequence_collector[seq][0] for seq in self.population_loc])

            idx_sort = np.argsort(self.fitness_loc)[::-1]
            print(f"    (Local) Top Fitness: {self.fitness_loc[idx_sort[0]]}")
            print(f"    (Local) Top Sequence: {self.population_loc[idx_sort[0]]}")

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

            # Exchange populations
            best_seq_local = population_sort[0 : self.num_exchanges]
            best_fitness_local = fitness_sort[0 : self.num_exchanges]

            idx_sort = np.argsort(self.fitness)[::-1]
            worst_indices = idx_sort[-self.num_exchanges :]
            for i, idx in enumerate(worst_indices):
                try:
                    self.population[idx] = best_seq_local[i]
                    self.fitness[idx] = best_fitness_local[i]
                except:
                    continue

            # Save best of generation
            fit_all_best = np.argmax(self.fitness)
            with open(
                os.path.join(self.work_dir, "generation_all_best.txt"), "a+"
            ) as f:
                f.writelines(
                    f"Gen:{gen_}, {self.population[fit_all_best]}, {self.fitness[fit_all_best]} \n"
                )

        return

    @staticmethod
    def get_good_bad_sequences(fitness, population, generation_size):
        """
        Split population into sequences to keep and sequences to replace
        based on fitness values.
        """
        fitness = np.array(fitness)
        idx_sort = fitness.argsort()[::-1]
        keep_ratio = 0.2
        keep_idx = math.ceil(len(list(idx_sort)) * keep_ratio)

        try:
            F_50_val = fitness[idx_sort[keep_idx]]
            F_25_val = np.array(fitness) - F_50_val
            F_25_val = np.array([x for x in F_25_val if x < 0]) + F_50_val
            F_25_sort = F_25_val.argsort()[::-1]
            F_25_val = F_25_val[F_25_sort[0]]

            prob_ = 1.0 / (3.0 ** ((F_50_val - fitness) / (F_50_val - F_25_val)) + 1)

            prob_ = prob_ / sum(prob_)
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

            if keep_seqs == [] or replace_seqs == []:
                raise Exception("Badly sampled population!")
        except:
            keep_seqs = [population[i] for i in idx_sort[:keep_idx]]
            replace_seqs = [population[i] for i in idx_sort[keep_idx:]]

        return keep_seqs, replace_seqs

    def log(self):
        pass

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]