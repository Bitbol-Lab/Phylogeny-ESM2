import os
import subprocess
from argparse import ArgumentParser

# This code runs main.py for all the families in the directory of the natural MSA given.
if __name__ == "__main__":
    # Parsing command-line options
    parser = ArgumentParser()

    # Set directory that contains natural MSA
    parser.add_argument("-f", "--msa_dir", action="store", dest="msa_dir",
                        help="write path to MSA directory", default="Data/MSA/")

    # Set directory that contains tree
    parser.add_argument("-t", "--tree_dir", action="store", dest="tree_dir",
                        help="write directory that contains or will contain trees", default='Data/Tree/')

    # Set tree method
    parser.add_argument("-m", "--method", action="store", dest="method",
                        default='FastTree',
                        help="write the method to use. Options are: FastTree or IQTree")

    # Set batch_size
    parser.add_argument("-b", "--batch_size", action="store", dest="batch_size",
                        default='1', help="write the batch size")

    # Set model
    parser.add_argument("-e", "--model", action="store", dest="model",
                        default="facebook/esm2_t6_8M_UR50D", help="write the ESM model to use from Hugging Face")

    # Set tree to be computed
    parser.add_argument("-c", "--tree_to_compute", action="store_false", dest="tree_to_compute",
                        default=True, help="write True if the tree needs to be computed, false otherwise")

    # Set output directory
    parser.add_argument("-o", "--output", action="store", dest="output",
                        default="Data/Output/", metavar="FILE", help="directory where to save output")

    args = parser.parse_args()
    msa_dir = args.msa_dir
    tree_dir = args.tree_dir
    method = args.method
    batch_size = args.batch_size
    model = args.model
    tree_to_compute = args.tree_to_compute
    output_dir = args.output

    for filename in os.listdir(msa_dir):  # For all natural MSAs in the given directory
        f = os.path.join(msa_dir, filename)  # Get the path of the MSA file
        if os.path.isfile(f):
            if f.endswith('.fasta'):  # REMEMBER TO PUT THE RIGHT EXTENSION
                directory, filename = os.path.split(f)
                filename_no_extension = os.path.splitext(filename)[0]
                if tree_to_compute:  # If the tree needs to be computed
                    subprocess.run(["python", "main.py", "-f", f, "-d", tree_dir, "-m", method, "-b", batch_size,
                                    "-e", model, "-o", output_dir])
                else:  # If the tree is already given
                    tree_path = tree_dir + method + '/' + filename_no_extension
                    if method == 'FastTree':
                        tree_path += '.newick'
                    elif method == 'IQTree':
                        tree_path += '.treefile'
                    subprocess.run(["python", "main.py", "-f", f, "-t", tree_path, "-b", batch_size,
                                    "-e", model, "-o", output_dir])
                print(f'{f} finished')
