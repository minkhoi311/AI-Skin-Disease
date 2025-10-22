# AI-Skin-Disease
Ứng dụng **Trí tuệ nhân tạo (AI)** sử dụng **mô hình học sâu (Deep Learning)** — cụ thể là **Convolutional Neural Network (CNN)** để nhận dạng **bệnh lý da liễu** từ hình ảnh.  
Người dùng có thể tải ảnh vùng da và mô tả triệu chứng, hệ thống sẽ **dự đoán loại bệnh tương ứng** cùng **một số lời khuyên hữu ích**.

# Cài đặt và chạy
**Ngôn ngữ:** Python 3.11  
**Framework:** TensorFlow, Flask, scikit-learn

### 🔹 Clone repository
```bash
git clone https://github.com/minkhoi311/AI-Skin-Disease
cd AI-Skin-Disease
```

## Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

## Chạy chương trình giao diện (GUI)
```bash
python app.py
```

# Kiểm thử & Hiệu năng
- Thời gian dự đoán 1 ảnh: ~2.3s 
- Độ chính xác trung bình: 60~90% trên tập kiểm thử
- Kết hợp CNN + NLP 

# License & Compatibility
- License: MIT License (OSI-approved)
- Thư viện sử dụng:
TensorFlow (Apache 2.0),
NumPy (BSD),
Matplotlib (PSF),
scikit-learn (BSD),

# Liên hệ & Góp ý
- GitHub Issues: https://github.com/minkhoi311/AI-Skin-Disease/issues
- Email: - 2351050126nhu@ou.edu.vn
       - 2351050084khoi@ou.edu.vn
       - 2351050210yen@ou.edu.vn

# Nhóm thực hiện
Sinh viên Trường Đại học Mở Thành phố Hồ Chí Minh
- Lê Minh Khôi 
- Nguyễn Trần Quỳnh Như
- Lê Bảo Yến
- Mục tiêu: Phát triển hệ thống hỗ trợ chẩn đoán về bệnh da tự động.