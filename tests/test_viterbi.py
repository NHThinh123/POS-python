import unittest
from src.viterbi_tagger import ViterbiPOSTagger

class TestViterbiPOSTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = ViterbiPOSTagger()
        self.tagger.train()

    def test_tag_known_sentence(self):
        result = self.tagger.tag("The dog runs.")
        self.assertEqual(len(result), 4)  # Bao gồm dấu chấm
        self.assertEqual(result[0][1], "DT")

if __name__ == "__main__":
    unittest.main()