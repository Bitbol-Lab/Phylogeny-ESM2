import numpy as np
from abc import ABC, abstractmethod


class MSAGenerator(ABC):
    """
    Class that generates an MSA based on a single sequence
    """

    def __init__(self, number_of_nodes, number_state_spin):
        """
        Constructor method
        """
        # Set the number of nodes as the length of the sequence
        self.number_of_nodes = number_of_nodes
        # Set the number of states 20 amino acids
        # spin = index of the possible state
        self.number_state_spin = number_state_spin

    @abstractmethod
    def msa_tree_phylo(self, clade_root, flip_before_start, first_sequence, neff=1.0):
        pass

    @abstractmethod
    def mcmc(self, number_of_mutation, l_spin):
        pass

    def msa_tree_phylo_recur(self, clade_root, previous_sequence, msa, neff):
        """
        Recurrent function to create the MSA for a given tree
        :param clade_root:
        :param previous_sequence: initial sequence to be mutated
        :param msa: MSA of the sequences in the leafs of the phylogenetic tree
        :param neff: number of mutations per site per branch length
        :return: modified MSA
        """
        # Define the new sequence as a vector with the same length of the previous sequence
        new_sequence = np.zeros((previous_sequence.shape[0]), dtype=np.int8)

        # Obtain the node of the tree
        b = clade_root.clades
        if len(b) > 0:  # If b is not a leaf
            for clade in b:
                # Start with the previous sequence (copied in order to not modify the previous sequence)
                new_sequence[:] = previous_sequence
                # Compute number of mutations
                n_mutations = int(clade.branch_length * new_sequence.shape[0] * neff)
                # Create new sequence with the given number of mutations
                self.mcmc(n_mutations, new_sequence)
                # Recursive step
                self.msa_tree_phylo_recur(clade, new_sequence, msa, neff)
        else:  # If b is a leaf
            # Compute number of mutations
            n_mutations = int(clade_root.branch_length * previous_sequence.shape[0] * neff)
            # Create new sequence with the given number of mutations (sequence is not copied because base case)
            self.mcmc(n_mutations, previous_sequence)
            # Save the leaf sequence in the MSA
            msa[int(clade_root.name), :] = previous_sequence
            print(clade_root.name)
        return msa
