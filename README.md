# AI-Skin-Disease
Dự án này sử dụng mô hình học sâu (Deep Learning) — cụ thể là Convolutional Neural Network (CNN) để nhận dạng các bệnh lý da liễu từ hình ảnh.
Người dùng có thể cung cấp ảnh và mô tả triệu chứng của mình, chương trình sẽ dự đoán loại bệnh tương ứng cùng một số lời khuyên.

# Cài đặt và chạy
Ngôn ngữ lập trình: Python 3.11s
1 Clone repository
git clone https://github.com/minkhoi311/AI-Skin-Disease
cd AI-Skin-Disease

2 Cài đặt thư viện cần thiết
pip install -r requirements.txt

3 Chạy chương trình giao diện (GUI)
python test1.py

4 Hoặc chạy dự đoán nhanh qua terminal
python test1.py images.jpg

# Kiểm thử & Hiệu năng
Thời gian dự đoán 1 ảnh: ~2.3s trên CPU Intel i5-10400
Độ chính xác trung bình: 60~90% trên tập kiểm thử
Kết hợp CNN + NLP 

# License & Compatibility
License: MIT License (OSI-approved)
Thư viện sử dụng:
TensorFlow (Apache 2.0)
NumPy (BSD)
Matplotlib (PSF)
scikit-learn (BSD)
PyQt5 (GPLv3)
Googletrans (MIT)
=> Tất cả đều tương thích với MIT License

# Liên hệ & Góp ý
GitHub Issues: https://github.com/minkhoi311/AI-Skin-Disease/issues
Email: 2351050126nhu@ou.edu.vn
       2351050084khoi@ou.edu.vn
       2351050210yen@ou.edu.vn

# Nhóm thực hiện
Sinh viên Trường Đại học Mở Thành phố Hồ Chí Minh
- Lê Minh Khôi 
- Nguyễn Trần Quỳnh Như
- Lê Bảo Yến
Mục tiêu: Phát triển hệ thống hỗ trợ chẩn đoán về bệnh da tự động.