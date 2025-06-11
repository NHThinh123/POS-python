POS Tagger Project
Dự án triển khai 4 thuật toán gán nhãn POS: MostFreqPOStagger, ViterbiPOStagger, CRFPOStagger, MEMMPOStagger.
Cài đặt

Cài Python 3.11 từ https://www.python.org/downloads/.
Tạo môi trường ảo:python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


Cài thư viện:pip install -r requirements.txt


Tải dữ liệu NLTK:python setup_nltk.py



Sử dụng
Chạy chương trình chính:
python main.py

Chạy bài kiểm tra:
python -m unittest discover tests

Cấu trúc thư mục

data/: Lưu trữ dữ liệu (nếu cần).
src/: Code các tagger.
tests/: Bài kiểm tra đơn vị.
requirements.txt: Danh sách thư viện.
main.py: Chạy thử nghiệm tất cả tagger.

