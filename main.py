from MSAGenerator.MSAGeneratorESM import MSAGeneratorESM
from Bio import SeqIO, Phylo
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
from argparse import ArgumentParser
import numpy as np
import time
import os
import matplotlib.pyplot as plt


def construct_tree(path_msa, dir_tree, tree_method):
    """
    Construct three from given MSA.
    :param path_msa: path to MSA.
    :param dir_tree: directory to save trees.
    :param tree_method: method to construct tree.
    :return: constructed tree.
    """
    file_directory, file_name = os.path.split(path_msa)
    filename_without_extension = os.path.splitext(file_name)[0]

    if path_msa.endswith('.seed'):  # Read the MSA in stockholm format and rewrite in fasta format
        natural_msa = SeqIO.parse(path_msa, "stockholm")
        new_filename = filename_without_extension + ".fasta"
        path_msa = os.path.join(file_directory, new_filename)
        SeqIO.write(natural_msa, path_msa, "fasta")

    if tree_method == 'FastTree':  # Build the tree using FastTree
        command = ["FastTree", path_msa]
        os.makedirs(dir_tree + 'FastTree/', exist_ok=True)
        path_tree = dir_tree + 'FastTree/' + filename_without_extension + '.newick'
        with open(path_tree, "w") as outfile:
            subprocess.call(command, stdout=outfile)
        constructed_tree = Phylo.read(path_tree, "newick")
    elif tree_method == 'IQTree':  # Build the tree using IQTree
        os.makedirs(dir_tree + 'IQTree/', exist_ok=True)
        path_tree = dir_tree + 'IQTree/' + filename_without_extension
        command = ["iqtree", "-s", path_msa, "--prefix", path_tree]
        subprocess.call(command)
        constructed_tree = Phylo.read(path_tree + ".treefile", "newick")
    else:
        raise ValueError("Wrong method name")
    return constructed_tree


if __name__ == "__main__":
    # Start time
    start_time = time.time()

    # Parsing command-line options
    parser = ArgumentParser()
    # Path to the MSA in fasta or stockholm format
    parser.add_argument("-f", "--msa_path", action="store", dest="msa_path",
                        help="write path to MSA", metavar="FILE")

    # Path to the tree, if already present
    parser.add_argument("-t", "--tree_path", action="store", dest="tree_path",
                        help="write path to tree if tree already computed", metavar="FILE")

    # Path to the tree directory, where to save the trees
    parser.add_argument("-d", "--tree_dir", action="store", dest="tree_dir",
                        help="write path to tree directory if tree to be computed", default="Data/Tree/",
                        metavar="FILE")

    # Set tree method
    parser.add_argument("-m", "--method", action="store", dest="method",
                        default='FastTree', help="write the method to use. Options are: FastTree or IQTree")

    # Set batch_size
    parser.add_argument("-b", "--batch_size", action="store", dest="batch_size",
                        default='1', help="write the batch size", type=int)

    # Set model
    parser.add_argument("-e", "--model", action="store", dest="model",
                        default="facebook/esm2_t6_8M_UR50D", help="write the ESM model to use from Hugging Face")

    # Set output directory
    parser.add_argument("-o", "--output", action="store", dest="output",
                        default="Data/Output/", metavar="FILE", help="directory where to save output")

    # Get the arguments
    args = parser.parse_args()
    msa_path = args.msa_path
    tree_path = args.tree_path
    tree_dir = args.tree_dir
    method = args.method
    batch_size = args.batch_size
    model = args.model
    output_dir = args.output

    # Construct and/or read the tree
    if tree_path is None:  # If the path to the tree is not given, the tree is computed
        tree = construct_tree(msa_path, tree_dir, method)
    else:  # If the path to the tree is given, read the MSA and the tree
        tree = Phylo.read(tree_path, "newick")

    plt.rc('font', size=6)
    # set the size of the figure
    fig = plt.figure(figsize=(10, 20), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes)
    plt.show()

    # Read MSA
    if msa_path.endswith('.seed'):
        msa = SeqIO.parse(msa_path, "stockholm")
    else:
        msa = SeqIO.parse(msa_path, "fasta")

    # Get the sequence with the minimum number of gaps and
    # get the association between sequences' ids and position in MSA
    dict_seq_id = {}  # Dictionary containing the association between record id and position in the MSA
    minimum = -1
    sequence = ""
    for index, record in enumerate(msa):
        dict_seq_id[record.id] = index
        current_sequence = str(record.seq)
        number_of_gaps = current_sequence.count('-')
        if minimum < 0 or number_of_gaps < minimum:
            minimum = number_of_gaps
            sequence = current_sequence
    # Remove all the gaps in the sequence
    sequence_without_gaps = sequence.replace('-', "")

    # Rename the leaf of the tree, so that the order of sequences in the synthetic MSA follows the order
    # of the natural MSA
    for leaf in tree.root.get_terminals():
        index = dict_seq_id[leaf.name]
        leaf.name = str(index)

    # Generate the new MSA using the constructed tree
    msa_generator = MSAGeneratorESM(len(sequence_without_gaps), 20, batch_size=batch_size, model=model)
    final_msa = msa_generator.msa_tree_phylo(tree.root, 0, sequence_without_gaps)

    # Save the MSA in numpy format
    directory, filename = os.path.split(msa_path)
    filename_no_extension = os.path.splitext(filename)[0]
    model_parameters = model.split('_')[-2]
    output_path = os.path.join(output_dir, model_parameters)
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path + '/' + filename_no_extension + method + '.npy', final_msa)

    # Save MSA in fasta format
    seq_records = []
    for i in range(final_msa.shape[0]):
        seq_records.append(SeqRecord(seq=Seq(''.join(msa_generator.inverse_amino_acid_map[index]
                                                     for index in final_msa[i])),
                                     id='seq' + str(i)))
    SeqIO.write(seq_records, output_path + '/' + filename_no_extension + method + '.fasta', "fasta")

    # End time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
