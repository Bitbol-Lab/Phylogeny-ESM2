from argparse import ArgumentParser
import matplotlib.pyplot as plt
from Bio import Phylo, AlignIO
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os
from transformers import AutoTokenizer
import subprocess

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
amino_acid_map = {char: index for index, char in enumerate(tokenizer.all_tokens) if char in
                  ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                   '-', 'X']}

families = [
    'PF00004',
    'PF00005',
    'PF00041',
    'PF00072',
    'PF00076',
    'PF00096',
    'PF00153',
    'PF00271',
    'PF00397',
    'PF00512',
    'PF00595',
    'PF01535',
    'PF02518',
    'PF07679',
    'PF13354'
]


if __name__ == "__main__":
    # Parsing command-line options
    parser = ArgumentParser()
    # Path to the MSA in fasta or stockholm format
    parser.add_argument("-f", "--msa_dir", action="store", dest="msa_dir",
                        help="path to MSA directory", default=None, metavar="FILE")

    # Path to the tree directory, where to save the trees
    parser.add_argument("-t", "--tree_dir", action="store", dest="tree_dir",
                        help="path to tree directory if tree to be printed", default=None, metavar="FILE")

    # Tree method
    parser.add_argument("-m", "--method", action="store", dest="method",
                        help="method that have been used. Options are: FastTree or IQTree", default=None)

    # MAFFT alignment flag
    parser.add_argument("-a", "--alignment", action="store_true", dest="alignment",
                        help="", default=False)

    # Output directory
    parser.add_argument("-o", "--output", action="store", dest="output",
                        help="path to output directory", default=None, metavar="FILE")

    # Get the arguments
    args = parser.parse_args()
    msa_dir = args.msa_dir
    tree_dir = args.tree_dir
    method = args.method
    output_dir = args.output
    compute_alignment = args.alignment

    if msa_dir is None or output_dir is None or method is None:
        print("Both MSA and Output directories and Method must be specified in the command arguments (-f, -o and -m)")
        exit()

    pearson_correlation_df = {
        'Family': [],
        'Correlation_All': [],
        'Correlation_Restrict': []
    }

    for family in families:  # for each family
        pearson_correlation_df['Family'].append(family)
        # read natural MSA and synthetic MSA
        msa_path = os.path.join(msa_dir, family + '.alignment.fasta')
        msa_synthetic_path = os.path.join(output_dir, family + f'.alignment{method}.fasta')

        if tree_dir is not None:  # If the directory of the tre is given, print the tree
            tree_path = os.path.join(tree_dir, method + '/' + family)
            if method == 'FastTree':
                tree_path += '.alignment.newick'
            elif method == 'IQTree':
                tree_path += '.alignment.treefile'

            tree = Phylo.read(tree_path, "newick")
            msa = AlignIO.read(msa_path, "fasta")

            plt.rc('font', size=6)
            # set the size of the figure
            fig = plt.figure(figsize=(10, 20), dpi=100)
            # alternatively
            # fig.set_size_inches(10, 20)
            axes = fig.add_subplot(1, 1, 1)
            Phylo.draw(tree, axes=axes)
            plt.show()

        # Read natural MSA and transform it into a numpy array
        msa = AlignIO.read(msa_path, "fasta")
        num_sequences = len(msa)
        alignment_length = msa.get_alignment_length()
        msa_natural = np.zeros((num_sequences, alignment_length))
        for index, record in enumerate(msa):
            msa_natural[index, :] = np.array([amino_acid_map[aa] for aa in record.seq if aa in amino_acid_map])

        # If flag is true, compute the alignment using mafft
        if compute_alignment:
            aligned_msa_synthetic_path = msa_synthetic_path.replace('.fasta', '.mafft.fasta')
            with open(aligned_msa_synthetic_path, 'w') as out:
                command = ["mafft", "--maxiterate", "1000", "--localpair", msa_synthetic_path]
                subprocess.run(command, stdout=out)
            msa_synthetic_path = aligned_msa_synthetic_path

        # Read synthetic MSA and transform it into a numpy array
        msa = AlignIO.read(msa_synthetic_path, "fasta")
        num_sequences = len(msa)
        alignment_length = msa.get_alignment_length()
        msa_synthetic = np.zeros((num_sequences, alignment_length))
        for index, record in enumerate(msa):
            msa_synthetic[index, :] = np.array([amino_acid_map[aa] for aa in record.seq if aa in amino_acid_map])

        # Compute the hamming distances between sequences of the natural MSA and sequences of synthetic MSA
        x = []
        y = []
        x_restrict = []
        y_restrict = []
        # For all the pairs of sequences
        for i in range(msa_synthetic.shape[0]):
            for j in range(i + 1, msa_synthetic.shape[0]):
                # Compute normalized hamming distance for sequences in natural MSA
                differences_synthetic = msa_synthetic[i] != msa_synthetic[j]
                x.append(np.sum(differences_synthetic) / msa_synthetic.shape[1])
                # Compute normalized hamming distance for sequences in synthetic MSA
                differences_natural = msa_natural[i] != msa_natural[j]
                y.append(np.sum(differences_natural) / msa_natural.shape[1])

                if y[-1] <= 0.9:
                    x_restrict.append(np.sum(differences_synthetic) / msa_synthetic.shape[1])
                    y_restrict.append(np.sum(differences_natural) / msa_natural.shape[1])

        # Compute pearson correlation between synthetic and natural distances
        pearson_correlation, _ = pearsonr(x, y)
        pearson_correlation_df['Correlation_All'].append(pearson_correlation)
        # Plot a graph of these distances
        plt.scatter(x=x, y=y)
        plt.title(family)
        plt.xlabel('Hamming distance in synthetic MSA')
        plt.ylabel('Hamming distance in natural MSA')
        plt.show()
        # Compute pearson correlation between synthetic and natural distances when natural distance is lower than 0.5
        pearson_correlation, _ = pearsonr(x_restrict, y_restrict)
        pearson_correlation_df['Correlation_Restrict'].append(pearson_correlation)
    df = pd.DataFrame(pearson_correlation_df)
    print(df)
