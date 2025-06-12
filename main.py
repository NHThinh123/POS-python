from src.most_freq_tagger import MostFreqPOSTagger
from src.viterbi_tagger import ViterbiPOSTagger
from src.crf_tagger import CRFPOSTagger
from src.memm_tagger import MEMMPOSTagger
from src.utils import evaluate_tagger
from nltk.corpus import treebank
import time


def main():
    test_percentage = 0.3
    # Nhập câu từ người dùng
    test_sentence = input("Nhập câu để gán nhãn: ")

    # Khởi tạo các tagger
    taggers = {
        "MostFreqPOSTagger": {
            "tagger": MostFreqPOSTagger(),
            "complexity": "Huấn luyện: O(N_w), Gán nhãn: O(L)"
        },
        "ViterbiPOSTagger": {
            "tagger": ViterbiPOSTagger(),
            "complexity": "Huấn luyện: O(N_t), Gán nhãn: O(L * T^2)"
        },
        "CRFPOSTagger": {
            "tagger": CRFPOSTagger(),
            "complexity": "Huấn luyện: O(N_t * T^2 * I), Gán nhãn: O(L * T^2)"
        },
        "MEMMPOSTagger": {
            "tagger": MEMMPOSTagger(),
            "complexity": "Huấn luyện: O(N_t * F * D), Gán nhãn: O(L * F)"
        }
    }

    # Chia dữ liệu
    tagged_sents = treebank.tagged_sents()
    total_sents = len(tagged_sents)
    test_size = int(total_sents * test_percentage)
    train_sents = tagged_sents[:-test_size]
    test_sents = tagged_sents[-test_size:]

    # Huấn luyện và đánh giá
    for name, info in taggers.items():
        tagger = info["tagger"]
        complexity = info["complexity"]
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
        print(f"Kết quả gán nhãn: {tagged}")

        # Đánh giá độ chính xác
        accuracy = evaluate_tagger(tagger, test_sents)
        error_rate = 1 - accuracy

        # Hiển thị hiệu suất
        print(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}")
        print(f"Tỷ lệ lỗi: {error_rate:.4f}")
        print(f"Thời gian huấn luyện: {train_time:.2f} giây")
        print(f"Thời gian gán nhãn: {tag_time:.4f} giây")
        print(f"Độ phức tạp tính toán: {complexity}")
        print(f"Trong đó: N_w = số từ, N_t = số token, L = độ dài câu, T = số nhãn, I = số vòng lặp, F = số đặc trưng, D = độ sâu cây")


if __name__ == "__main__":
    main()