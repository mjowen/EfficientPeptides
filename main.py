import os
import copy
import numpy as np
import re
import blosum

from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from Bio import SeqIO
from Bio.Seq import Seq

rng = np.random.default_rng()


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


def solve_strucutures(input_file, output_title):
    os.system(f"colabfold_batch /work/inputs/{input_file} /work/outputs/{output_title}")


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
    min_n_of_elites = 10
    # How many elites to keep
    if len(elites) - len(population) < min_n_of_elites: # Make sure we have at least `min_n_of_elites` elites
        population.sort(key=protein_cost)
        population = population[:len(elites)-min]
    elite_count = len(elites)-len(population)
    # Sort the population by cost
    elites.sort(key=protein_cost)
    for i in range(elite_count):
        population.append(elites[i])
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
    population = [copy.copy(canonical_seq) for _ in range(200)]
    write_fasta("data/initial_population.fasta", population)
    solve_strucutures("data/initial_population.fasta", "data/optimised")
    # Compare the structures to the canonical coords for the TM score distribution
    res = [align(file, f"data/pdb/gfp_{i}.pdb") for i in range(1, 200)]
    tm_threshold = np.percentile(res, 5) # Set the TM-score threshold to the 5th percentile
    for gen in range(1, 1000):
        print(f"Generation {gen}")
        # Save pop to pass elites
        print("Population size:", len(population))
        elites = copy.deepcopy(population)
        # Tournament selection
        newpop = tournament_selection(population)
        # Crossover
        newpop = crossover(newpop)
        # Mutate the sequences
        newpop = mutate(newpop, mask=[0,1,2,62,63,64,65,66,67,68,235,236,237]) # First amino acid, SYG chromophore, last amino acid, neighbouring 2 amino acids on each side
        write_fasta(f"/work/inputs/trail_population_{gen}.fasta", newpop) # TODO I dont need to check the constraint against sequences that I've already checked
        solve_strucutures(f"trial_population_{gen}.fasta", f"gen_{gen}")
        # Compare the structures to the canonical coords for the TM score distribution
        res = [align(file, f"/work/outputs/gen_{gen}/gfp_{i}.pdb") for i in range(1, 200)]
        print("TM Scores:", res)
        # Remove any sequences that are too structurally distinct from the canonical sequence
        newpop = [seq for seq, tm in zip(newpop, res) if tm > tm_threshold]
        # Add elites to fill population
        population = set_elites(newpop, elites)
    return population

optimise("/work/inputs/canonical.pdb")
# TODO add to dockerfile
# TODO add logging statements
