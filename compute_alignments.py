# This code is taken from the paper "Protein language models trained on multiple sequence alignments learn phylogenetic
# relationships"

import os
import string
import subprocess
from argparse import ArgumentParser
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq


def sequences_to_str_array(filename, fmt="fasta"):
    return np.asarray([list(str(record.seq)) for record in SeqIO.parse(filename, fmt)])


if __name__ == "__main__":
    # Parsing command-line options
    parser = ArgumentParser()
    # Path to the MSA in fasta or stockholm format
    parser.add_argument("-f", "--msa_dir", action="store", dest="msa_dir",
                        help="path to MSA directory", default=None, metavar="FILE")

    # Path to the tree directory, where to save the trees
    parser.add_argument("-m", "--hmm_dir", action="store", dest="hmm_dir",
                        help="path to hmm directory", default=None, metavar="FILE")

    parser.add_argument("-o", "--output_dir", action="store", dest="output_dir",
                        help="path to output directory", default=None, metavar="FILE")

    args = parser.parse_args()
    msa_dir = args.msa_dir
    hmm_dir = args.hmm_dir
    output_dir = args.output_dir

    for filename in os.listdir(msa_dir):  # For all natural MSAs in the given directory
        f = os.path.join(msa_dir, filename)  # Get the path of the MSA file
        if os.path.isfile(f):
            if f.endswith('.seed'):
                print(f)
                # 1) Get path for seed MSAs and PFAM HMMs
                file_directory, file_name = os.path.split(f)
                filename_without_extension = os.path.splitext(file_name)[0]
                hmm_file = hmm_dir + filename.split('.')[0] + '.hmm'

                # 2) Align seed MSAs to downloaded HMMs with hmmalign, producing Stockholm file
                output_file = output_dir + filename.split('.')[0] + 'alignment_hmm.seed'
                subprocess.run(["hmmalign", "--amino", "-o", output_file, hmm_file, f])

                # 3) Convert Stockholm aligned MSA to FASTA format
                output_file_fasta = output_dir + filename.split('.')[0] + 'alignment_hmm.fasta'
                parsed = list(SeqIO.parse(output_file, "stockholm"))
                with open(output_file_fasta, "w") as output_handle:
                    SeqIO.write(parsed, output_handle, "fasta")

                # 4) Keep only match and deletion states
                msa_arr = sequences_to_str_array(output_file_fasta)
                valid_idxs = []
                for col in range(msa_arr.shape[1]):
                    unique_set = set(np.unique(msa_arr[:, col]))
                    if not set(string.ascii_lowercase).intersection(unique_set):
                        valid_idxs.append(col)

                no_inserts_filename = output_dir + filename.split('.')[0] + ".alignment.fasta"
                parsed = list(SeqIO.parse(output_file_fasta, "fasta"))
                with open(no_inserts_filename, "w") as output_handle:
                    for i, record in enumerate(parsed):
                        record.seq = Seq("".join(msa_arr[i, valid_idxs]))
                    SeqIO.write(parsed, output_handle, "fasta")

                os.remove(output_file)
                os.remove(output_file_fasta)
