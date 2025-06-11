from collections import defaultdict, Counter
from nltk.corpus import treebank
import math

class ViterbiPOSTagger:
    def __init__(self):
        self.trans_prob = defaultdict(lambda: defaultdict(float))
        self.emit_prob = defaultdict(lambda: defaultdict(float))
        self.tag_counts = Counter()
        self.tags = set()

    def train(self, train_sents=None):
        if train_sents is None:
            train_sents = treebank.tagged_sents()
        trans_counts = defaultdict(Counter)
        emit_counts = defaultdict(Counter)
        for sent in train_sents:
            prev_tag = '<START>'
            for word, tag in sent:
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                prev_tag = tag
            trans_counts[prev_tag]['<END>'] += 1
        for prev_tag in trans_counts:
            total = sum(trans_counts[prev_tag].values())
            for tag in trans_counts[prev_tag]:
                self.trans_prob[prev_tag][tag] = trans_counts[prev_tag][tag] / total
        for tag in emit_counts:
            total = sum(emit_counts[tag].values())
            for word in emit_counts[tag]:
                self.emit_prob[tag][word] = emit_counts[tag][word] / total
        self.default_tag = max(self.tag_counts, key=self.tag_counts.get)

    def tag(self, input_data):
        if isinstance(input_data, str):
            from nltk.tokenize import word_tokenize
            words = word_tokenize(input_data)
        else:
            words = input_data
        n = len(words)
        V = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]
        for tag in self.tags:
            V[0][tag] = math.log(self.trans_prob['<START>'].get(tag, 1e-6)) + math.log(self.emit_prob[tag].get(words[0], 1e-6))
            backpointer[0][tag] = None
        for i in range(1, n):
            for tag in self.tags:
                max_prob, max_prev_tag = float('-inf'), None
                for prev_tag in self.tags:
                    prob = V[i-1][prev_tag] + math.log(self.trans_prob[prev_tag].get(tag, 1e-6)) + math.log(self.emit_prob[tag].get(words[i], 1e-6))
                    if prob > max_prob:
                        max_prob, max_prev_tag = prob, prev_tag
                V[i][tag] = max_prob
                backpointer[i][tag] = max_prev_tag
        max_prob, best_tag = float('-inf'), None
        for tag in self.tags:
            prob = V[n-1][tag] + math.log(self.trans_prob[tag].get('<END>', 1e-6))
            if prob > max_prob:
                max_prob, best_tag = prob, tag
        tags = [best_tag]
        for i in range(n-1, 0, -1):
            best_tag = backpointer[i][best_tag]
            tags.append(best_tag)
        tags.reverse()
        return list(zip(words, tags))