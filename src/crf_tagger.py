from nltk.corpus import treebank
import pycrfsuite

class CRFPOSTagger:
    def __init__(self):
        self.tagger = pycrfsuite.Tagger()
        self.model_path = 'pos_crf.model'

    def _word2features(self, sent, i, prev_tags=None):
        word = sent[i]
        features = [
            f'word.lower={word.lower()}',
            f'word.isupper={word.isupper()}',
            f'word.istitle={word.istitle()}',
            f'word.isdigit={word.isdigit()}',
            f'word.len={len(word)}',
            f'word.suffix2={word[-2:].lower() if len(word) >= 2 else ""}',
            f'word.suffix3={word[-3:].lower() if len(word) >= 3 else ""}',
        ]
        if i > 0:
            word1 = sent[i-1]
            features.extend([
                f'-1:word.lower={word1.lower()}',
                f'-1:word.istitle={word1.istitle()}',
                f'-1:word.isupper={word1.isupper()}',
                f'-1:word.len={len(word1)}',
            ])
            if prev_tags and i > 0:
                features.append(f'-1:tag={prev_tags[i-1]}')
        else:
            features.append('BOS')
        if i > 1:
            word2 = sent[i-2]
            features.extend([
                f'-2:word.lower={word2.lower()}',
                f'-2:word.istitle={word2.istitle()}',
            ])
        if i < len(sent)-1:
            word1 = sent[i+1]
            features.extend([
                f'+1:word.lower={word1.lower()}',
                f'+1:word.istitle={word1.istitle()}',
                f'+1:word.isupper={word1.isupper()}',
                f'+1:word.len={len(word1)}',
            ])
        if i < len(sent)-2:
            word2 = sent[i+2]
            features.extend([
                f'+2:word.lower={word2.lower()}',
                f'+2:word.istitle={word2.istitle()}',
            ])
        else:
            features.append('EOS')
        return features

    def _sent2features(self, sent, prev_tags=None):
        return [self._word2features(sent, i, prev_tags) for i in range(len(sent))]

    def _sent2labels(self, sent):
        return [tag for _, tag in sent]

    def train(self, train_sents=None):
        if train_sents is None:
            train_sents = treebank.tagged_sents()
        X_train = [self._sent2features([word for word, _ in sent]) for sent in train_sents]
        y_train = [self._sent2labels(sent) for sent in train_sents]
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
            'c1': 0.1,  # Giảm L1 regularization
            'c2': 0.01,  # Giảm L2 regularization
            'max_iterations': 200,  # Tăng số lần lặp
            'feature.possible_transitions': True
        })
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.train(self.model_path)
        self.tagger.open(self.model_path)

    def tag(self, input_data):
        if isinstance(input_data, str):
            from nltk.tokenize import word_tokenize
            words = word_tokenize(input_data)
        else:
            words = input_data
        features = self._sent2features(words)
        tags = self.tagger.tag(features)
        return list(zip(words, tags))