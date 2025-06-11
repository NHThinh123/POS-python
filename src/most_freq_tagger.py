from collections import defaultdict, Counter
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize

class MostFreqPOSTagger:
    def __init__(self):
        self.word_to_most_freq_tag = {}
        self.default_tag = None

    def train(self, train_sents=None):
        if train_sents is None:
            train_sents = treebank.tagged_sents()
        word_tag_freq = defaultdict(Counter)
        total_tag_freq = Counter()
        for sent in train_sents:
            for word, tag in sent:
                word_tag_freq[word][tag] += 1
                total_tag_freq[tag] += 1
        self.word_to_most_freq_tag = {
            word: max(tags, key=tags.get) for word, tags in word_tag_freq.items()
        }
        self.default_tag = total_tag_freq.most_common(1)[0][0]

    def tag(self, input_data):
        if isinstance(input_data, str):
            words = word_tokenize(input_data)
        else:
            words = input_data
        return [(word, self.word_to_most_freq_tag.get(word, self.default_tag)) for word in words]