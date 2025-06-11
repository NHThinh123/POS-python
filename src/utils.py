from nltk.corpus import treebank

def evaluate_tagger(tagger, test_sentences):
    correct = 0
    total = 0
    for sent in test_sentences:
        words = [word for word, _ in sent]
        true_tags = [tag for _, tag in sent]
        predicted_tags = [tag for _, tag in tagger.tag(words)]
        if len(true_tags) != len(predicted_tags):
            print(f"Câu không khớp: {' '.join(words)}")
            continue
        for true, pred in zip(true_tags, predicted_tags):
            if true == pred:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Correct: {correct}, Total: {total}")
    return accuracy