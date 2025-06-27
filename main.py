import os
import copy
import numpy as np
import re
import blosum
import time

from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from Bio import SeqIO
from Bio.Seq import Seq

rng = np.random.default_rng()

INPUT_FILE = os.environ.get("INPUT_FILE", "/work/inputs/canonical.pdb")
if not os.path.isfile(INPUT_FILE):
    raise ValueError(f"INPUT_FILE '{INPUT_FILE}' does not exist or is not a valid file.")

POPULATION_SIZE = int(os.environ.get("POPULATION_SIZE", 200))
if POPULATION_SIZE <= 0 or POPULATION_SIZE % 2 != 0:
    raise ValueError("POPULATION_SIZE must be a positive even number.")

NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", 1000))
if NUM_GENERATIONS <= 0:
    raise ValueError("NUM_GENERATIONS must be a positive integer.")

MIN_NUM_ELITES = int(os.environ.get("MIN_NUM_ELITES", 10))
if MIN_NUM_ELITES < 0:
    raise ValueError("MIN_NUM_ELITES must be a non-negative integer.")

TM_THRESHOLD = int(os.environ.get("TM_THRESHOLD", 5))
if TM_THRESHOLD < 0 or TM_THRESHOLD > 100:
    raise ValueError("TM_THRESHOLD must be between 0 and 100.")

MASK_POSITIONS = os.environ.get("MASK_POSITIONS", "")
MASK_POSITIONS = [int(pos) for pos in MASK_POSITIONS.split(",") if pos]


def align(file1, file2):
    chain1 = next(get_structure(file1).get_chains())
    chain2 = next(get_structure(file2).get_chains())
    coords1, seq1 = get_residue_data(chain1)
    coords2, seq2 = get_residue_data(chain2)
    res = tm_align(coords1, coords2, seq1, seq2)
    return res.tm_norm_chain1


def get_seq_and_coords(file):
    """Get sequence and coordinates from a PDB file.
    Args:
        file (str): Path to the input PDB file.
    Returns:
        tuple: A tuple containing the coordinates and sequence.
    """
    structure = get_structure(file)
    chain = next(structure.get_chains())
    coords, seq = get_residue_data(chain)
    return coords, seq


def write_fasta(file, seqs):
    """Write a sequence to a FASTA file.
    Args:
        file (str): Path to the output FASTA file.
        seqs (list): Sequences to write.
    """
    records = [SeqIO.SeqRecord(Seq(seq.upper().replace("-","")), id=str(i)) for i, seq in enumerate(seqs)]
    with open(file, 'w') as f:
        SeqIO.write(records, f, "fasta")


def solve_structures(input_file, output_title):
    if "initial_population" in input_file:
        for i in range(POPULATION_SIZE):
            os.system(f"colabfold_batch --num-models 1 --seed {i} /work/inputs/{input_file} /work/outputs/{output_title}")
            # Rename model files
            os.system(f"mv /work/outputs/{output_title}/*_alphafold2_ptm_model_1_seed_{i:03}.pdb /work/outputs/{output_title}/predicted_{i}.pdb")
    else:
        os.system(f"colabfold_batch --num-models 1 /work/inputs/{input_file} /work/outputs/{output_title}")
        # Rename model files
        for i in range(POPULATION_SIZE):
            os.system(f"mv /work/outputs/{output_title}/{i}*_alphafold2_ptm_model_1*.pdb /work/outputs/{output_title}/predicted_{i}.pdb")
    # Rename files and delete unnecessary files
    os.system(f"rm /work/outputs/{output_title}/*.a3m")
    os.system(f"rm /work/outputs/{output_title}/*.json")
    os.system(f"rm -rd /work/outputs/{output_title}/*__env")
    os.system(f"rm -rd /work/outputs/{output_title}/*.png")
    os.system(f"rm -rd /work/outputs/{output_title}/*.done.txt")


def protein_cost(seq):
    """Calculate the cost of a protein sequence.
    Args:
        seq (str): Input protein sequence.
    Returns:
        float: Cost of the protein sequence.
    """
    atp_cost = {"A": -1, "R": 0, "N": 2, "D": 0,
                "C": 8, "E": -7, "Q": -6, "G": -2,
                "H": 3, "I": 7, "L": -9, "K": 5,
                "M": 18, "F": 0, "P": -2, "S": -2, 
                "T": 6, "W": -1, "Y": -2, "V": -2}
    yields = {"A": 2, "R": 1, "N": 1.73, "D": 2.0,
                "C": 1.24, "E": 1, "Q": 1, "G": 2,
                "H": 0.75, "I": 0.79, "L": 0.67, "K": 0.84,
                "M": 0.71, "F": 0.57, "P": 1, "S": 2, 
                "T": 1.37, "W": 0.44, "Y": 0.57, "V": 1.0}
    cost = 0
    for aa in seq:
        if aa == "-":
            continue
        cost += 26/yields[aa.upper()] + min(atp_cost[aa.upper()],0) + 4.2 # Opportunity cost due to consumed glucose + ATP consumed/released [consumed already accounted for in yields] + Polymerisation cost
    return cost


def tournament_selection(population):
    """Select sequences from the population using tournament selection.
    Deterministic tournement, size 2, without replacement.
    Args:
        population (list): List of sequences in the population.
    Returns:
        list: Selected sequences after tournament selection.
    """
    newpop = []
    for _ in range(2):
        rng.shuffle(population)
        for i in range(0, len(population), 2):
            seq1 = population[i]
            seq2 = population[i + 1]
            # Compare the sequences based on their cost
            cost1 = protein_cost(seq1)
            cost2 = protein_cost(seq2)
            if cost1 < cost2:
                newpop.append(seq1)
            else:
                newpop.append(seq2)
    return newpop


def crossover(population):
    """Perform crossover on the population of sequences.
    Args:
        population (list): List of sequences in the population.
    Returns:
        list: New population after crossover.
    """
    rng.shuffle(population)
    for i in range(0, len(population), 2):
        seq1 = population[i]
        seq2 = population[i + 1]
        # Perform crossover at a random point
        # Split the sequences into parts based on uppercase amino acids
        new_seq1 = re.split(r"(?=[A-Z\-])", seq1)
        new_seq2 = re.split(r"(?=[A-Z\-])", seq2)
        crossover_point = rng.integers(1, len(new_seq1)) # Random crossover point
        new_seq1 = new_seq1[:crossover_point] + new_seq2[crossover_point:]
        new_seq2 = new_seq2[:crossover_point] + new_seq1[crossover_point:]
        population[i] = "".join(new_seq1)
        population[i + 1] = "".join(new_seq2)
    return population

# Remove amino acid groups from the matrix
matrix = blosum.BLOSUM(62)
for aas in matrix.keys():
    for aa in ["B", "J", "Z", "X"]:
        del matrix[aas][aa]
for aa in ["B", "J", "Z", "X", "*"]:
    del matrix[aa]

def mutate(population, mask=[]):
    """Mutate a population of sequences using the BLOSUM62 substitution matrix.
    Args:
        seq (str): Input sequence to mutate.
    Returns:    
        str: Mutated sequence.
    """
    newpop = []
    for seq in population:
        newseq = ""
        canonical_counter = 0
        for aa in seq:
            # If it is masked, move on to the next amino acid
            if aa.isupper() or aa == "-":
                if canonical_counter in mask:
                    newseq += aa
                    canonical_counter += 1
                    continue
                canonical_counter += 1
            # If it is a deletion, move on to the next amino acid
            if aa == "-":
                newseq += "-"
                continue
            weights = np.array([2**i for i in matrix[aa.upper()].values()])
            weights = weights / np.sum(weights)
            if aa.islower():
                new_aa = np.random.choice(list(matrix[aa.upper()].keys()), p=weights).lower()
            else:
                new_aa = np.random.choice(list(matrix[aa].keys()), p=weights)
            if new_aa != "*":
                newseq += new_aa
            else:
                # Insertion or deletion
                if np.random.rand() < 0.1:  # 10% chance to insert
                    newseq += aa  # Keep the original amino acid
                    # Insert random amino acid after, lowercase to indicate insertion
                    newseq += np.random.choice(list(matrix.keys())).lower()
                else:
                    # Deletion is still marked by a dash if it is canonical
                    if aa.isupper():
                        newseq += "-"
        newpop.append(newseq)
    return newpop


def set_elites(population, elites):
    """Add elites until the population is full (min. `min_n_of_elites` elites, replacing worst if necessary).
    Args:
        population (list): List of sequences in the population.
        elites (list): List of elite sequences to add to the population.
    Returns:
        list: New population with elites added.
    """
    # How many elites to keep
    if len(elites) - len(population) < MIN_NUM_ELITES: # Make sure we have at least `min_n_of_elites` elites
        population.sort(key=protein_cost)
        population = population[:len(elites) - MIN_NUM_ELITES]
    elite_count = len(elites)-len(population)
    # Sort the population by cost
    elites.sort(key=protein_cost)
    print("Adding the following elites")
    for i in range(elite_count):
        population.append(elites[i])
        print(elites[i])
    return population


def optimise(file):
    """Optimise the structure in the given file.
    Args:
        file (str): Path to the input sequence file, e.g. a .fasta file.
    Returns:
        str: Path to the output file containing the optimised structure (.pdb).
    """
    # Get the TM-score threshold
    _, canonical_seq = get_seq_and_coords(file)
    population = [copy.copy(canonical_seq) for _ in range(POPULATION_SIZE)]
    write_fasta("/work/inputs/initial_population.fasta", [population[0]])
    solve_structures("initial_population.fasta", "gen_-1")
    print("Mask positions:", MASK_POSITIONS)
    # Compare the structures to the canonical coords for the TM score distribution
    res = [align(file, f"/work/outputs/gen_-1/predicted_{i}.pdb") for i in range(POPULATION_SIZE)]
    print(f"TM Scores for initial population:")
    for i, tm in enumerate(res):
        print(f"Predicted {i}: {tm:.3f}")
    tm_threshold = np.percentile(res, TM_THRESHOLD) # Set the TM-score threshold to the 5th percentile
    print(f"TM score threshold for selection: {tm_threshold:.3f}")
    for gen in range(1, NUM_GENERATIONS):
        # Log time
        start_time = time.time()
        print(f"Generation {gen}")
        print(f"Population size: {len(population)}")
        print(f"Number of unique individuals at start of generation: {len(set(population))}")
        print(f"Protein cost: {np.mean([protein_cost(seq) for seq in population]):.2f} ± {np.std([protein_cost(seq) for seq in population]):.2f}")

        # Save pop to pass elites
        elites = copy.deepcopy(population)

        # Tournament selection
        newpop = tournament_selection(population)

        # Crossover
        newpop = crossover(newpop)

        # Mutate the sequences
        newpop = mutate(newpop, mask=MASK_POSITIONS)

        print(f"Number of unique individuals before solving structures: {len(set(newpop))}")
        write_fasta(f"/work/inputs/trial_population_{gen}.fasta", newpop)
        solve_structures(f"trial_population_{gen}.fasta", f"gen_{gen}")

        # Compare the structures to the canonical coords for the TM score distribution
        res = [align(file, f"/work/outputs/gen_{gen}/predicted_{i}.pdb") for i in range(POPULATION_SIZE)]
        print(f"TM Scores for generation {gen}:")
        for i, tm in enumerate(res):
            print(f"Predicted {i}: {tm:.3f}")
        print(f"Removing {sum([tm <= tm_threshold for tm in res])} sequences with TM score below threshold {tm_threshold:.3f}")
        # Remove any sequences that are too structurally distinct from the canonical sequence
        newpop = [seq for seq, tm in zip(newpop, res) if tm > tm_threshold]

        # Add elites to fill population
        population = set_elites(newpop, elites)
        print(f"Time to compute generation {gen}: {time.time() - start_time:.2f} seconds")
    print("Printing final population")
    for i in range(len(population)):
        print(f"Sequence {i}: {population[i]}")
    return population

optimise(INPUT_FILE)
