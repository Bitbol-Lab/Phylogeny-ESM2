from argparse import ArgumentParser
import subprocess
import os


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
    # Tree method
    parser.add_argument("-m", "--method", action="store", dest="method",
                        help="method that have been used. Options are: FastTree or IQTree", default="")

    # HMM directory
    parser.add_argument("-s", "--hmm_dir", action="store", dest="hmm_dir",
                        help="hmm directory", default=None)

    # Output directory
    parser.add_argument("-o", "--output", action="store", dest="output",
                        help="path to synthetic msa directory", default=None, metavar="FILE")

    # Results directory
    parser.add_argument("-r", "--results", action="store", dest="results",
                        help="path to directory where to save results", default=None, metavar="FILE")

    # Get the arguments
    args = parser.parse_args()
    method = args.method
    hmm_dir = args.hmm_dir
    output_dir = args.output
    results_dir = args.results

    if output_dir is None or method is None:
        print("Output directories and Method must be specified in the command arguments (-o and -m)")
        exit()

    os.makedirs(hmm_dir, exist_ok=True)

    for family in families:  # for each family
        msa_synthetic_path = os.path.join(output_dir, family + f'.alignment{method}.fasta')

        command = ["hmmsearch",
                   "--tblout", os.path.join(results_dir, f"synthetic_scores_{family}.tbl"),
                   hmm_dir + f"{family}.hmm", msa_synthetic_path]
        subprocess.run(command)
