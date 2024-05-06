import numpy as np
import random
from MSAGenerator import MSAGenerator


class MSAGeneratorPottsModel(MSAGenerator):
    """
    Class that generates an MSA based on a single sequence using Potts Model
    """
    def __init__(self, field, coupling):
        """
        Constructor method
        :param field: parameter that can be inferred from data by DCA methods
        :param coupling: parameters that can be inferred from data by DCA methods
        """
        super().__init__(np.intc(field.shape[0]), np.intc(field.shape[1]))
        # Set fields and couplings
        self.field = field
        self.coupling = coupling

    def msa_tree_phylo(self, clade_root, flip_before_start, first_sequence, neff=1.0):
        """
        Initialize the MSA and start the recursion to compute node sequences
        :param clade_root: root of the tree
        :param flip_before_start: number of mutations to apply to the random generated sequence
        :param first_sequence: first sequence (root)
        :param neff: number of mutations per site per branch length
        :return: MSA of the sequences in the leafs of the phylogenetic tree
        """
        # Create a synthetic first sequence
        first_sequence = np.random.randint(0, high=self.number_state_spin, size=self.number_of_nodes,
                                           dtype=np.int8)
        # Initialize the MSA
        msa = np.zeros((len(clade_root.get_terminals()), self.number_of_nodes), dtype=np.int8)
        # Compute the first sequence (root)
        self.mcmc(flip_before_start, first_sequence)
        # Create the new sequences in the MSA recursively
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
            new_state = np.random.randint(0, self.number_state_spin - 1)
            # Avoid to select the same state as before
            if new_state >= l_spin[selected_node]:
                new_state += 1

            # Compute the difference in the value of H before and after the mutation (there is no - in the exp later)
            de = (self.pseudo_hamiltonian(selected_node, new_state, l_spin)
                  - self.pseudo_hamiltonian(selected_node, l_spin[selected_node], l_spin))

            # If the difference is positive or if it is greater than a random value, apply the mutation
            if de >= 0 or random.random() < np.exp(de):
                # Modify the selected position with the new selected state
                l_spin[selected_node] = new_state
                # Increase the number of mutation applied
                c_mutation += 1

    def pseudo_hamiltonian(self, node, state_node, l_spin):
        """
        Compute the pseudo Hamiltonian for computing the differences between Hamiltonian
        :param node: selected position in the sequence to be mutated
        :param state_node: state (amino acid) to consider for the mutation
        :param l_spin: sequence to be mutated
        :return: pseudo Hamiltonian
        """
        # Initialize the pseudo hamiltonian
        hamiltonian = self.field[node, state_node] - self.coupling[node, node, state_node, l_spin[node]]
        # Compute the summation of coupling between the position to mutate and all the positions
        # of the sequence (except the considered one, because it was subtracted before)
        for i in range(l_spin.shape[0]):
            hamiltonian += self.coupling[node, i, state_node, l_spin[i]]
        return hamiltonian

