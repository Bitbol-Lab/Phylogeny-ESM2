from MSAGenerator import MSAGenerator
import torch
import numpy as np
import pandas as pd
import random
from transformers import pipeline
from datasets import Dataset


class MSAGeneratorESM(MSAGenerator):
    """
    Class that generates an MSA based on a single sequence using ESM
    """

    def __init__(self, number_of_nodes, number_state_spin):
        """
        Constructor method
        :param number_of_nodes: length of the sequence
        :param number_state_spin: number of states (20 amino acids + 1 gap)
        """
        super().__init__(number_of_nodes, number_state_spin)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # Set the model
        # Load model directly
        self.pipe = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D", device=device)

        # Set the map for computing the sequences
        self.amino_acid_map = {char: index for index, char in enumerate(self.pipe.tokenizer.all_tokens) if char in
                               ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                                'H', 'W', 'C']}

        self.inverse_amino_acid_map = {v: k for k, v in self.amino_acid_map.items()}

    def transform_sequence_to_array(self, sequence):
        """
        Transform a string of characters in an array of integer
        :param sequence: string of characters representing the protein sequence
        :return: array of integer representing the protein sequence
        """
        return np.array([self.amino_acid_map[aa] for aa in sequence if aa in self.amino_acid_map])

    def transform_array_to_sequence(self, array):
        """
        Transform an array of integer in a string of characters
        :param array: array of integer representing the protein sequence
        :return: string of characters representing the protein sequence
        """
        return ''.join([self.inverse_amino_acid_map[val] for val in array])

    def msa_tree_phylo(self, clade_root, flip_before_start, first_sequence, neff=1.0):
        """
        Initialize the MSA and start the recursion to compute node sequences
        :param clade_root: root of the tree
        :param flip_before_start: number of mutations to apply to the random generated sequence
        :param first_sequence: first sequence (root)
        :param neff: number of mutations per site per branch length
        :return: MSA of the sequences in the leafs of the phylogenetic tree
        """
        # Initialize the root
        first_sequence = self.transform_sequence_to_array(first_sequence)
        # Initialize the MSA
        msa = np.zeros((len(clade_root.get_terminals()), self.number_of_nodes), dtype=np.int8)
        # Create the new sequences recursively
        return np.asarray(self.msa_tree_phylo_recur(clade_root, first_sequence, msa, neff))

    def mcmc(self, number_of_mutation, l_spin):
        """
        Apply to the given sequence the given number of mutations
        :param number_of_mutation: given number of mutations
        :param l_spin: given sequence
        :return: modified sequence
        """
        n = 64

        # Set the number of mutations
        c_mutation = 0

        # Until the number of mutation is not achieved
        while c_mutation < number_of_mutation:
            c_mutations1 = c_mutation
            # Select one position to mutate in the sequence
            selected_nodes = np.random.choice(np.arange(self.number_of_nodes), size=n, replace=True)

            # Select a new state
            new_states = np.random.randint(low=min(self.amino_acid_map.values()),
                                           high=max(self.amino_acid_map.values()) - 1, size=n)

            # Avoid to select the same state as before
            for i in range(n):
                if new_states[i] >= l_spin[selected_nodes[i]]:
                    new_states[i] += 1

            protein_sequence = self.transform_array_to_sequence(l_spin)

            masked_sequences = []
            # Mask the position
            for selected_node in selected_nodes:
                masked_sequence = protein_sequence[:selected_node] + "<mask>" + protein_sequence[selected_node + 1:]
                masked_sequences.append(masked_sequence)

            self.pipe.model.eval()
            with torch.no_grad():
                dataset = Dataset.from_dict({"sequences": masked_sequences})
                # Get the probabilities
                predictions = self.pipe(dataset['sequences'], top_k=33)

            for i in range(n):
                prediction = predictions[i]
                prediction = pd.DataFrame(prediction)
                score_old = prediction.loc[prediction['token'] == l_spin[selected_nodes[i]], 'score'].values
                score_new = prediction.loc[prediction['token'] == new_states[i], 'score'].values

                if len(score_old) > 0:
                    p_old = float(score_old[0])
                else:
                    raise ValueError

                if len(score_new) > 0:
                    p_new = float(score_new[0])
                else:
                    raise ValueError

                if len(score_new) > 1 or len(score_old) > 1:
                    print(predictions)

                # Compute the ratio between probabilities
                p = p_new / p_old

                # If the difference is positive or if it is greater than a random value, apply the mutation
                if random.random() < p or p >= 1:
                    # Modify the selected position with the new selected state
                    l_spin[selected_nodes[i]] = new_states[i]
                    # Increase the number of mutation applied
                    c_mutation += 1
                    break

            if c_mutations1 == c_mutation:
                print("Muori")
