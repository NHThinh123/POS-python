from nltk.corpus import treebank
from sklearn.feature_extraction import DictVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm

class MEMMPOSTagger:
    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.classifier = LGBMClassifier(
            n_estimators=200,  # Tăng số cây
            learning_rate=0.05,  # Giảm learning rate
            max_depth=10,  # Tăng độ sâu cây
            min_child_samples=10,  # Giảm ngưỡng mẫu
            n_jobs=-1,
            random_state=42,
            verbose=-1,  # Tắt log chi tiết
            class_weight='balanced'  # Cân bằng lớp
        )
        self.tags = []

    def _word2features(self, sent, i, prev_tag):
        word = sent[i]
        features = {
            'word.lower': word.lower(),
            'word.istitle': word.istitle(),
            'word.suffix2': word[-2:].lower() if len(word) >= 2 else '',
            'word.suffix3': word[-3:].lower() if len(word) >= 3 else '',
            'word.prefix2': word[:2].lower() if len(word) >= 2 else '',
            'word.prefix3': word[:3].lower() if len(word) >= 3 else '',
            'prev_tag': prev_tag,
            'word.isalpha': word.isalpha(),
            'is_plural': word.lower().endswith('s'),
            'position': i,  # Vị trí từ trong câu
            'prev_tag+suffix2': f"{prev_tag}_{word[-2:].lower()}" if len(word) >= 2 else prev_tag,
        }
        if i > 0:
            word1 = sent[i-1]
            features.update({
                '-1:word.lower': word1.lower(),
                '-1:word.istitle': word1.istitle(),
            })
        if i > 1:
            word2 = sent[i-2]
            features.update({
                '-2:word.lower': word2.lower(),
                '-2:word.istitle': word2.istitle(),
            })
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            word1 = sent[i+1]
            features.update({
                '+1:word.lower': word1.lower(),
                '+1:word.istitle': word1.istitle(),
            })
        if i < len(sent)-2:
            word2 = sent[i+2]
            features.update({
                '+2:word.lower': word2.lower(),
                '+2:word.istitle': word2.istitle(),
            })
        else:
            features['EOS'] = True
        return features

    def train(self, train_sents=None):
        if train_sents is None:
            train_sents = treebank.tagged_sents()
        X, y = [], []
        self.tags = sorted(set(tag for sent in train_sents for _, tag in sent))
        for sent in train_sents:
            prev_tag = '<START>'
            for i, (word, tag) in enumerate(sent):
                features = self._word2features([w for w, _ in sent], i, prev_tag)
                X.append(features)
                y.append(tag)
                prev_tag = tag
        X_vectorized = self.vectorizer.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.1, random_state=42)
        self.classifier.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lightgbm.early_stopping(stopping_rounds=10, verbose=False)]
        )
        return self.classifier

    def tag(self, input_data):
        if isinstance(input_data, str):
            from nltk.tokenize import word_tokenize
            words = word_tokenize(input_data)
        else:
            words = input_data
        tags = []
        prev_tag = '<START>'
        for i in range(len(words)):
            features = self._word2features(words, i, prev_tag)
            X = self.vectorizer.transform([features])
            probs = self.classifier.predict_proba(X)[0]
            tag = self.classifier.predict(X)[0]
            tags.append(tag)
            prev_tag = tag
        return list(zip(words, tags))