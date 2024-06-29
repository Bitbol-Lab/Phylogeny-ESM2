import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function copied from StackOverflow
def read_hmmer(path, program="hmmsearch", format="tblout", add_header_as_index=False, verbose=True):
    if format in {"tblout","domtblout"}:
        cut_index = {"tblout":18, "domtblout":22}[format]
        data = list()
        header = list()
        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    header.append(line)
                else:
                    row = list(filter(bool, line.strip().split(" ")))
                    row = row[:cut_index] + [" ".join(row[cut_index:])]
                    data.append(row)

        df = pd.DataFrame(data)
        if not df.empty:
            if format == "tblout":
                columns = list(map(lambda field: ("identifier", field), ["target_name", "target_accession", "query_name", "query_accession"])) \
                        + list(map(lambda field: ("full_sequence", field), ["e-value", "score", "bias"])) \
                        + list(map(lambda field: ("best_domain", field), ["e-value", "score", "bias"])) \
                        + list(map(lambda field: ("domain_number_estimation", field), ["exp", "reg", "clu",  "ov", "env", "dom", "rep", "inc"])) \
                        + list(map(lambda field: ("identifier", field), ["query_description"]))
            if format == "domtblout":
                columns = list(map(lambda field: ("identifier", field), ["target_name", "target_accession", "target_length", "query_name", "query_accession", "query_length"])) \
                        + list(map(lambda field: ("full_sequence", field), ["e-value", "score", "bias"])) \
                        + list(map(lambda field: ("this_domain", field), ["domain_number", "total_domains", "e-value","i-value", "score", "bias"])) \
                        + list(map(lambda field: ("hmm_coord", field), ["from", "to"])) \
                        + list(map(lambda field: ("ali_coord", field), ["from", "to"])) \
                        + list(map(lambda field: ("env_coord", field), ["from", "to"])) \
                        + list(map(lambda field: ("identifier", field), ["acc", "query_description"]))

            df.columns = pd.MultiIndex.from_tuples(columns)
    if add_header_as_index:
        df.index.name = "\n".join(map(str.strip, np.asarray(header)[[5,9,11]]))
    return df


# Pfam families with their MSA depths
families = {
    'PF00004': 207,
    'PF00005': 55,
    'PF00041': 98,
    'PF00072': 52,
    'PF00076': 68,
    'PF00096': 159,
    'PF00153': 160,
    'PF00271': 420,
    'PF00397': 422,
    'PF00512': 265,
    'PF00595': 44,
    'PF01535': 458,
    'PF02518': 658,
    'PF07679': 46,
    'PF13354': 66
}


def get_hmmer_scores(path):
    """Calculates percentage of sequences that pass the HMMER search and returns their scores."""
    rows = []
    # Iterate over all families
    for family, tot_sequences_num in families.items():
        # Read the HMMER search results from file
        hmm_scores = read_hmmer(os.path.join(path, f'synthetic_scores_{family}.tbl'))

        # Calculate the percentage of sequences that passed the HMMER search
        sequences_num = hmm_scores.shape[0]
        percentage = sequences_num / tot_sequences_num

        # If no sequence was found by HMMER search, column 'full_sequences' is not defined
        scores = hmm_scores['full_sequence']['score'].astype(float).to_numpy() if 'full_sequence' in hmm_scores else []

        # Create the row
        rows.append([family, sequences_num, tot_sequences_num, percentage, scores])

    # Create and sort dataframe
    df = pd.DataFrame(rows, columns=["Family", "HMMER Sequences", "Tot Sequences", "Percentage", "HMMER Scores"])
    df.sort_values('Percentage', ascending=False, inplace=True)
    return df


if __name__ == '__main__':
    # Parsing command-line options
    parser = ArgumentParser()
    # Path to the HMMER search results for synthetic MSAs
    parser.add_argument(action="store", dest="hmm_search_dir",
                        help="path to HMMER search results directory for synthetic MSAs", default=None, metavar="SYNTHETIC_DIR")

    # Path to the HMMER search results for natural MSAs
    parser.add_argument(action="store", dest="hmm_natural_search_dir",
                        help="path to HMMER search results directory for natural MSAs", default=None, metavar="NATURAL_DIR")

    # Get the arguments
    args = parser.parse_args()
    hmm_search_dir = args.hmm_search_dir
    hmm_natural_search_dir = args.hmm_natural_search_dir

    # Load HMMER scores for both synthetic and natural MSAs
    synthetic_hmmer_scores = get_hmmer_scores(hmm_search_dir)
    natural_hmmer_scores = get_hmmer_scores(hmm_natural_search_dir)

    # Print the percentage of sequences passing the HMMER search
    print("Synthetic:")
    print(synthetic_hmmer_scores.drop('HMMER Scores', axis=1).head(15))

    print("\nNatural:")
    print(natural_hmmer_scores.drop('HMMER Scores', axis=1).head(15))

    # Plot HMMER scores of synthetic and natural MSAs
    for i in range(3):
        data_synthetic = synthetic_hmmer_scores.iloc[i]['HMMER Scores']
        data_natural = natural_hmmer_scores.iloc[i]['HMMER Scores']

        pos = [1, 1.7]
        fig, ax = plt.subplots()
        ax.violinplot([data_synthetic, data_natural], pos)
        ax.set_ylabel("HMMER Scores")
        ax.set_xticks(pos, ["Synthetic", "Natural"])
        fig.show()
