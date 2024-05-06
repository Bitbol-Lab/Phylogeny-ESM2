from MSAGenerator import MSAGenerator
import torch
import numpy as np
import pandas as pd
import random
from transformers import pipeline


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
        # Set the model
        self.pipe = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")
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
        # Set the parameters
        selected_node, new_state, c_mutation = 0, 0, 0

        # Until the number of mutation is not achieved
        while c_mutation < number_of_mutation:
            # Select one position to mutate in the sequence
            selected_node = np.random.randint(0, self.number_of_nodes)

            # Select a new state
            new_state = np.random.randint(min(self.amino_acid_map.values()), max(self.amino_acid_map.values()) - 1)

            # Avoid to select the same state as before
            if new_state >= l_spin[selected_node]:
                new_state += 1

            protein_sequence = self.transform_array_to_sequence(l_spin)
            # Mask the position
            masked_sequence = protein_sequence[:selected_node] + "<mask>" + protein_sequence[selected_node + 1:]

            self.pipe.model.eval()

            with torch.no_grad():
                # Get the probabilities
                predictions = pd.DataFrame(self.pipe(masked_sequence, top_k=33))

            score_old = predictions.loc[predictions['token'] == l_spin[selected_node], 'score'].values
            score_new = predictions.loc[predictions['token'] == new_state, 'score'].values

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
                l_spin[selected_node] = new_state
                # Increase the number of mutation applied
                c_mutation += 1
