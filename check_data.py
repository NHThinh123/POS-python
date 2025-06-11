from nltk.corpus import treebank

tagged_sents = treebank.tagged_sents()
print(f"Tổng số câu: {len(tagged_sents)}")
print(f"Câu ví dụ: {tagged_sents[0]}")