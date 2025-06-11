import unittest
from src.most_freq_tagger import MostFreqPOSTagger

class TestMostFreqPOSTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = MostFreqPOSTagger()
        self.tagger.train()

    def test_tag_known_word(self):
        result = self.tagger.tag("the")
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0][1], "DT")

    def test_tag_unknown_word(self):
        result = self.tagger.tag("unknownword")
        self.assertEqual(result[0][1], self.tagger.default_tag)

if __name__ == "__main__":
    unittest.main()