import unittest

from MSAGenerator.MSAGeneratorESM import MSAGeneratorESM
from Bio import Phylo
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import torch.nn as nn


def modify_leafs_names(tree):
    """
    Draw the tree and modifies leafs' names it in place.
    :param tree: tree to be modified.
    :return: modified tree.
    """
    # Draw the tree
    Phylo.draw(tree, do_show=False)
    plt.show()

    # Rename the names of leafs in the tree
    index = 0
    for leaf in tree.root.get_terminals():
        leaf.name = str(index)
        index += 1


class Test(unittest.TestCase):
    """
    Testing of the generation of sequences by the MSAGeneratorESM class.
    """
    def setUp(self):
        """
        Sets up the test.
        """
        # Set the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

        # Set the pipeline for comparison with the tokenizer
        self.pipe = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")

        # Set the amino_acid_map
        self.amino_acid_map = {char: index for index, char in enumerate(self.tokenizer.all_tokens) if char in
                               ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                                'H', 'W', 'C']}
        self.inverse_amino_acid_map = {v: k for k, v in self.amino_acid_map.items()}

        # Set the softmax
        self.softmax = nn.Softmax(dim=0)

    #######
    # These functions are used by the tests
    #######

    def compute_msa(self, size, tree, batch_size):
        """
        Compute final MSA of sequences.
        :param size: size of the sequences.
        :param tree: tree.
        :param batch_size: batch size.
        :return: random sequence and final MSA generated from the tree.
        """
        # Generate a random sequence
        random_sequence = np.random.randint(low=min(self.amino_acid_map.values()),
                                            high=max(self.amino_acid_map.values()) - 1, size=size)
        random_sequence_char = ''.join(self.inverse_amino_acid_map[i] for i in random_sequence)

        # Generate the MSA Generator
        msa_generator = MSAGeneratorESM(size, 20, batch_size=batch_size, model="facebook/esm2_t6_8M_UR50D")
        final_msa = msa_generator.msa_tree_phylo(tree.root, 0, random_sequence_char)
        return random_sequence, final_msa

    def check_probabilities(self, sequence):
        """
        Check the probabilities of masked positions
        :param sequence: sequence to be considered
        """
        # Select a random node
        selected_nodes = np.random.choice(np.arange(len(sequence)), size=1, replace=True)
        # Masked the sequence at that node
        masked_sequence = sequence[:selected_nodes[0]] + '<mask>' + sequence[selected_nodes[0] + 1:]
        # Obtain the tokenized sequence
        tokenized_sequence = self.tokenizer(masked_sequence, return_tensors="pt")
        b_input_ids = tokenized_sequence['input_ids']
        b_input_mask = tokenized_sequence['attention_mask']
        # Get the logits and the probabilities
        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        predictions_ESM = self.softmax(logits[0, selected_nodes[0] + 1, :])

        # Get the probabilities also from the pipeline
        predictions_pipeline = self.pipe(masked_sequence)

        # Evaluate that the prediction from the model are the same as the pipeline
        for value in predictions_pipeline:
            token = value['token']
            score = value['score']
            self.assertEqual(score, predictions_ESM[token])

    #####
    # These first tests are used to see if the generation along the tree works correctly
    #####
    def test_generator_along_tree_1(self):
        """
        Check if the generation along a given tree works correctly.
        """
        # Read the tree
        tree = Phylo.read("test_tree/Tree.newick", "newick")
        modify_leafs_names(tree)

        # Generate MSA
        size = 5
        batch_size = 2
        random_sequence, final_msa = self.compute_msa(size, tree, batch_size)

        for sequence in final_msa:
            self.assertEqual(np.sum(sequence != random_sequence), 1)

    def test_generator_along_tree_2(self):
        """
        Check if the generation along a given tree works correctly
        """
        # Read the tree
        tree = Phylo.read("test_tree/Tree1.newick", "newick")
        modify_leafs_names(tree)

        # Generate a random sequence with a random size
        size = 5
        batch_size = 6
        random_sequence, final_msa = self.compute_msa(size, tree, batch_size)

        for index in range(3):
            if index < 2:
                self.assertLessEqual(np.sum(final_msa[index] != random_sequence), 2)
                self.assertGreaterEqual(np.sum(final_msa[index] != random_sequence), 1)
            else:
                self.assertEqual(np.sum(final_msa[index] != random_sequence), 1)

    def test_generator_along_tree_3(self):
        """
        Check if the generation along a given tree works correctly.
        """
        # Read the tree
        tree = Phylo.read("test_tree/Tree2.newick", "newick")
        modify_leafs_names(tree)

        # Generate a random sequence with a random size
        size = 5
        batch_size = 4
        random_sequence, final_msa = self.compute_msa(size, tree, batch_size)

        for value in final_msa:
            self.assertEqual(np.sum(value != random_sequence), 1)

    #####
    # These second tests are used to check that the probabilities generated by the model and the pipeline are the same
    #####
    def test_generator_probabilities_1(self):
        """
        Check that the probabilities generated by the model and the pipeline are the same.
        """
        sequence = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
        for i in range(100):
            self.check_probabilities(sequence)

    def test_generator_probabilities_2(self):
        """
        Check that the probabilities generated by the model and the pipeline are the same.
        """
        sequence = 'AAAAA'
        for i in range(5):
            self.check_probabilities(sequence)

    #####
    # These third tests are used to check the correct mutation of sequences
    #####
    def test_mcmc(self):
        """
        Check that sequences are mutated correctly.
        """
        for i in range(10):
            # Generate a random sequence of random size
            size = np.random.randint(low=5, high=50)
            random_sequence = np.random.randint(low=min(self.amino_acid_map.values()),
                                                high=max(self.amino_acid_map.values()) - 1, size=size)

            # Generate a sequence from the random generated sequence with a single mutation
            previous_sequence = np.zeros(size)
            previous_sequence[:] = random_sequence[:]
            msa_generator = MSAGeneratorESM(size, 20, batch_size=1, model="facebook/esm2_t6_8M_UR50D")
            mutation = 1
            msa_generator.mcmc(mutation, random_sequence)

            # Check that the two sequences are differing by only one amino acid
            self.assertEqual(np.sum(previous_sequence != random_sequence), 1)

            # Generate another sequence from the random generated sequence with two mutations
            random_sequence[:] = previous_sequence[:]
            msa_generator = MSAGeneratorESM(size, 20, batch_size=2, model="facebook/esm2_t6_8M_UR50D")
            mutation = 2
            msa_generator.mcmc(mutation, random_sequence)

            # Check that the two sequences are differing by at least one and at most two amino acids
            self.assertGreaterEqual(np.sum(previous_sequence != random_sequence), 1)
            self.assertLessEqual(np.sum(previous_sequence != random_sequence), 2)
