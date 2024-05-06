from MSAGeneratorESM import MSAGeneratorESM
from Bio import SeqIO, Phylo
import subprocess
from optparse import OptionParser
import numpy as np
import time


if __name__ == "__main__":
    start_time = time.time()

    # Parsing command-line options
    parser = OptionParser()
    # Path to the MSA in fasta format
    parser.add_option("-f", "--file", action="store", dest="msa_path", type='string',
                      default='MSA/sample.fasta',
                      help="write path to MSA", metavar="FILE")

    # Path to first sequence in fasta format
    parser.add_option("-s", "--sequence", action="store", dest="sequence_path", type='string',
                      default='Sequence/P38126.fasta',
                      help="write path to first SEQUENCE", metavar="FILE")

    # Path to the tree, if already present
    parser.add_option("-t", "--tree_path", action="store", dest="tree_path", type='string',
                      help="write path to tree", metavar="FILE")

    # Set tree method
    parser.add_option("-m", "--method", action="store", dest="method", type='string',
                      default='FastTree',
                      help="write path where to store the obtained tree", metavar="FILE")

    (options, args) = parser.parse_args()
    msa_path = options.msa_path
    sequence_path = options.sequence_path
    tree_path = options.tree_path
    method = options.method

    if tree_path is None:
        # Build the tree using FastTree
        if method == 'FastTree':
            command = ["./FastTree", msa_path]
            tree_path = 'output_tree.newick'
            with open(tree_path, "w") as outfile:
                subprocess.call(command, stdout=outfile)
            tree = Phylo.read(tree_path, "newick")

        elif method == 'IQTree':
            command = ["iqtree", "-s", msa_path]
            subprocess.call(command)
            tree = Phylo.read(msa_path + ".treefile", "newick")

        else:
            raise ValueError("Wrong method name")
    else:
        tree = Phylo.read(tree_path, "newick")

    index = 0
    print(len(tree.root.get_terminals()))
    for leaf in tree.root.get_terminals():
        leaf.name = str(index)
        index += 1

    sequence = str(SeqIO.read(sequence_path, "fasta").seq)

    msa_generator = MSAGeneratorESM(len(sequence), 20)
    final_msa = msa_generator.msa_tree_phylo(tree.root, 0, sequence)
    np.save('msa.npy', final_msa)
    # Your program or code here
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
