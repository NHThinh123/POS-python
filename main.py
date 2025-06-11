from src.most_freq_tagger import MostFreqPOSTagger
from src.viterbi_tagger import ViterbiPOSTagger
from src.crf_tagger import CRFPOSTagger
from src.memm_tagger import MEMMPOSTagger
from src.utils import evaluate_tagger
from nltk.corpus import treebank
import time


def main():
    test_percentage = 0.2
    test_sentence = "The dog runs."

    # Khởi tạo các tagger
    taggers = {
        "MostFreqPOStagger": MostFreqPOSTagger(),
        "ViterbiPOStagger": ViterbiPOSTagger(),
        "CRFPOStagger": CRFPOSTagger(),
        "MEMMPOStagger": MEMMPOSTagger()
    }

    # Chia dữ liệu
    tagged_sents = treebank.tagged_sents()
    total_sents = len(tagged_sents)
    test_size = int(total_sents * test_percentage)
    train_sents = tagged_sents[:-test_size]
    test_sents = tagged_sents[-test_size:]

    # Huấn luyện và đánh giá
    for name, tagger in taggers.items():
        print(f"\nĐang huấn luyện {name}...")

        # Đo thời gian huấn luyện
        start_train = time.time()
        tagger.train(train_sents)
        train_time = time.time() - start_train
        print(f"Hoàn tất huấn luyện {name}.")

        print(f"Câu: {test_sentence}")

        # Đo thời gian gán nhãn
        start_tag = time.time()
        tagged = tagger.tag(test_sentence)
        tag_time = time.time() - start_tag
        print(f"Kết quả gán nhãn ({name}): {tagged}")

        # Đánh giá độ chính xác
        accuracy = evaluate_tagger(tagger, test_sents)

        # Hiển thị hiệu suất
        print(f"Độ chính xác trên tập kiểm tra ({name}): {accuracy:.4f}")
        print(f"Thời gian huấn luyện ({name}): {train_time:.2f} giây")
        print(f"Thời gian gán nhãn ({name}): {tag_time:.4f} giây")


if __name__ == "__main__":
    main()