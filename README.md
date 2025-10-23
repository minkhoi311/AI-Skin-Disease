# AI-Skin-Disease
Ứng dụng **Trí tuệ nhân tạo (AI)** sử dụng **mô hình học sâu (Deep Learning)** — cụ thể là **Convolutional Neural Network (CNN)** để nhận dạng **bệnh lý da liễu** từ hình ảnh.  
Người dùng có thể tải ảnh vùng da và mô tả triệu chứng, hệ thống sẽ **dự đoán loại bệnh tương ứng** cùng **một số lời khuyên hữu ích**.

# Cài đặt và chạy
**Ngôn ngữ:** Python 3.11  
**Framework:** TensorFlow, Flask, scikit-learn

### 🔹 Clone repository
#  AI-Skin-Disease — Hệ thống nhận diện bệnh da bằng AI

Dự án này sử dụng **Deep Learning (CNN)** kết hợp với **Xử lý ngôn ngữ tự nhiên (NLP)** để **nhận dạng các bệnh lý da liễu từ hình ảnh và mô tả triệu chứng**.

Người dùng có thể tải **ảnh da bị bệnh** và nhập **triệu chứng mô tả**, hệ thống sẽ **dự đoán loại bệnh**, đồng thời **đưa ra lời khuyên chăm sóc da phù hợp**.

---

## 1. Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.11
* **Thư viện chính:**

  * TensorFlow / Keras — huấn luyện và chạy mô hình CNN
  * Flask — xây dựng giao diện web
  * scikit-learn — xử lý văn bản (TF-IDF, cosine similarity)
  * pandas, numpy — xử lý dữ liệu
  

---

##  2. Cài đặt và chạy

### 🔹 Bước 1: Clone dự án


```bash
git clone https://github.com/minkhoi311/AI-Skin-Disease
cd AI-Skin-Disease
```

### 🔹 Bước 2: Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

#### Chạy chương trình giao diện (GUI)
```bash
python app.py
```
### 🔹 Bước 3: Chuẩn bị dữ liệu & mô hình

* Đặt mô hình đã huấn luyện (`cnn_model_final.h5`) vào thư mục gốc dự án.
* Tạo thư mục `Dataset/train` chứa các thư mục con tương ứng với từng loại bệnh (mỗi thư mục là một lớp).
* Đảm bảo file `skin_diseases.xlsx` có **2 cột bắt buộc**:


  * `Disease` — tên bệnh (trùng với tên thư mục trong Dataset)
  * `Symptom` — mô tả triệu chứng tương ứng



##  3. Chạy ứng dụng web

```bash
python app.py
```

Tại đây, bạn có thể:

* **Tải ảnh da liễu** từ máy tính
* **Nhập mô tả triệu chứng**
* Nhận **kết quả dự đoán bệnh** và **lời khuyên chăm sóc**

---

##  4. Mô hình và cơ chế kết hợp

Ứng dụng kết hợp:

* **CNN (ảnh)** → Nhận dạng trực quan bệnh lý.
* **TF-IDF + cosine similarity (văn bản)** → So khớp mô tả triệu chứng với cơ sở dữ liệu.


##  5. Hiệu năng

* **Thời gian dự đoán trung bình:** ~2–3 giây / ảnh
* **Độ chính xác (ước lượng):** 60–80% trên tập kiểm thử

---

##  6. License

* **Giấy phép:** 
  * MIT License
  * BSD License
  * Apache License 2.0
  * Python Software Foundation
* **Các thư viện sử dụng:**
  * TensorFlow (Apache 2.0)
  * NumPy (BSD)
  * scikit-learn (BSD)
  * Flask (BSD)
  * NumPy (BSD),
  * Matplotlib (PSF)
  * OpenPyXL (MIT)

---

##  7. Liên hệ & Góp ý

* GitHub Issues: [https://github.com/minkhoi311/AI-Skin-Disease/issues]
* Email:

  * [2351050126nhu@ou.edu.vn]
  * [2351050084khoi@ou.edu.vn]
  * [2351050210yen@ou.edu.vn]

---

##  8. Nhóm thực hiện

**Sinh viên Trường Đại học Mở Thành phố Hồ Chí Minh**

* Lê Minh Khôi
* Nguyễn Trần Quỳnh Như
* Lê Bảo Yến

**Mục tiêu:** Phát triển hệ thống web hỗ trợ **chẩn đoán bệnh da tự động**, thân thiện và hữu ích cho người dùng.
