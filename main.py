from MSAGeneratorESM import MSAGeneratorESM
from Bio import SeqIO, Phylo
import subprocess
from optparse import OptionParser
import numpy as np
import time
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Start time
    start_time = time.time()

    # Parsing command-line options
    parser = OptionParser()
    # Path to the MSA in fasta or stockholm format
    parser.add_option("-f", "--file", action="store", dest="msa_path", type='string',
                      default='MSA/PF00004.alignment.seed',
                      help="write path to MSA", metavar="FILE")

    # Path to the tree, if already present
    parser.add_option("-t", "--tree_path", action="store", dest="tree_path", type='string',
                      help="write path to tree", metavar="FILE")

    # Set tree method
    parser.add_option("-m", "--method", action="store", dest="method", type='string',
                      default='FastTree',
                      help="write path where to store the obtained tree", metavar="FILE")

    (options, args) = parser.parse_args()
    msa_path = options.msa_path
    tree_path = options.tree_path
    method = options.method

    if tree_path is None:
        directory, filename = os.path.split(msa_path)
        filename_no_extension = os.path.splitext(filename)[0]

        if msa_path.endswith('.seed'):
            msa = SeqIO.parse(msa_path, "stockholm")
            new_filename = filename_no_extension + ".fasta"
            msa_path = os.path.join(directory, new_filename)
            SeqIO.write(msa, msa_path, "fasta")
        else:
            msa = SeqIO.parse(msa_path, "fasta")

        # Build the tree using FastTree
        if method == 'FastTree':
            command = ["./FastTree", msa_path]
            tree_path = 'Tree/FastTree/' + filename_no_extension + '.newick'
            with open(tree_path, "w") as outfile:
                subprocess.call(command, stdout=outfile)
            tree = Phylo.read(tree_path, "newick")
        elif method == 'IQTree':
            tree_path = 'Tree/IQTree/' + filename_no_extension
            command = ["iqtree", "-s", msa_path, "--prefix", tree_path]
            subprocess.call(command)
            tree = Phylo.read(tree_path + ".treefile", "newick")

        else:
            raise ValueError("Wrong method name")
    else:
        # Get the sequence with the minimum number of gaps
        if msa_path.endswith('.seed'):
            msa = SeqIO.parse(msa_path, "stockholm")
        else:
            msa = SeqIO.parse(msa_path, "fasta")
        tree = Phylo.read(tree_path, "newick")

    index = 0
    for leaf in tree.root.get_terminals():
        leaf.name = str(index)
        index += 1
    print(index)

    plt.rc('font', size=6)
    # set the size of the figure
    fig = plt.figure(figsize=(10, 20), dpi=100)
    # alternatively
    # fig.set_size_inches(10, 20)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes)
    plt.show()

    minimum = -1
    sequence = ""
    for record in msa:
        current_sequence = str(record.seq)
        number_of_gaps = current_sequence.count('-')
        if minimum < 0 or number_of_gaps < minimum:
            minimum = number_of_gaps
            sequence = current_sequence
    sequence_without_gaps = sequence.replace('-', "")

    msa_generator = MSAGeneratorESM(len(sequence_without_gaps), 20)
    final_msa = msa_generator.msa_tree_phylo(tree.root, 0, sequence_without_gaps)
    directory, filename = os.path.split(msa_path)
    filename_no_extension = os.path.splitext(filename)[0]
    try:
        os.makedirs("Output")
    except OSError:
        # Handle the case where the directory already exists or there's a permission error
        pass
    np.save('Output/' + filename_no_extension + method + '.npy', final_msa)

    # End time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
