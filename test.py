import unittest

from MSAGeneratorESM import MSAGeneratorESM
from transformers import pipeline
from Bio import Phylo
import matplotlib.pyplot as plt


def count_different_chars(string1, string2):
    num_different = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            num_different += 1
    return num_different


class Test(unittest.TestCase):
    def setUp(self):
        self.sequence = "AAAAA"
        self.pipe = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")
        self.amino_acid_map = {char: index for index, char in enumerate(self.pipe.tokenizer.all_tokens) if char in
                               ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                                'H', 'W', 'C']}

        self.inverse_amino_acid_map = {v: k for k, v in self.amino_acid_map.items()}

    def test_generator(self):
        tree = Phylo.read("Test/Tree.newick", "newick")
        Phylo.draw(tree, do_show=False)
        plt.show()
        index = 0
        for leaf in tree.root.get_terminals():
            leaf.name = str(index)
            index += 1
        msa_generator = MSAGeneratorESM(len(self.sequence), 20)
        final_msa = msa_generator.msa_tree_phylo(tree.root, 0, self.sequence)
        for value in final_msa:
            result_sequence = ''.join(self.inverse_amino_acid_map[i] for i in value)
            self.assertEqual(count_different_chars(result_sequence, self.sequence), 1)

    def test_generator1(self):
        tree = Phylo.read("Test/Tree1.newick", "newick")
        Phylo.draw(tree, do_show=False)
        plt.show()
        index = 0
        for leaf in tree.root.get_terminals():
            leaf.name = str(index)
            index += 1
        msa_generator = MSAGeneratorESM(len(self.sequence), 20)
        final_msa = msa_generator.msa_tree_phylo(tree.root, 0, self.sequence)
        for index in range(3):
            if index < 2:
                result_sequence = ''.join(self.inverse_amino_acid_map[i] for i in final_msa[index])
                self.assertLessEqual(count_different_chars(result_sequence, self.sequence), 2)
            else:
                result_sequence = ''.join(self.inverse_amino_acid_map[i] for i in final_msa[index])
                self.assertEqual(count_different_chars(result_sequence, self.sequence), 1)

    def test_generator2(self):
        tree = Phylo.read("Test/Tree2.newick", "newick")
        Phylo.draw(tree, do_show=False)
        plt.show()
        index = 0
        for leaf in tree.root.get_terminals():
            leaf.name = str(index)
            index += 1
        msa_generator = MSAGeneratorESM(len(self.sequence), 20)
        final_msa = msa_generator.msa_tree_phylo(tree.root, 0, self.sequence)
        for value in final_msa:
            result_sequence = ''.join(self.inverse_amino_acid_map[i] for i in value)
            self.assertEqual(count_different_chars(result_sequence, self.sequence), 1)
