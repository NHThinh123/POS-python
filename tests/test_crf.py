import unittest
from src.crf_tagger import CRFPOSTagger

class TestCRFPOSTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = CRFPOSTagger()
        self.tagger.train()

    def test_tag_known_sentence(self):
        result = self.tagger.tag("The dog runs.")
        self.assertEqual(len(result), 4)

if __name__ == "__main__":
    unittest.main()