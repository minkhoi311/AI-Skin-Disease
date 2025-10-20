import sys
import os
import numpy as np
from googletrans import Translator
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==== 1. Cấu hình ban đầu ====
# Cố định seed để kết quả dự đoán không thay đổi mỗi lần
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)

MODEL_PATH = "cnn_model_final.h5"
IMG_SIZE = (192, 192)
DATASET_PATH = "Dataset/train"

# ==== 2. Tải danh sách lớp ====
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {DATASET_PATH}")
class_names = sorted(os.listdir(DATASET_PATH))

# ==== 3. Load model ====
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Lỗi khi tải model {MODEL_PATH}: {e}")

# ==== 4.5. Bản đồ nhãn tiếng Anh -> tiếng Việt ====
label_map = {
    "Acne and Rosacea Photos": "Mụn trứng cá và chứng đỏ mặt (Rosacea)",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Tổn thương ác tính và tiền ung thư da",
    "Atopic Dermatitis Photos": "Viêm da cơ địa",
    "Cellulitis Impetigo and other Bacterial Infections": "Nhiễm trùng da do vi khuẩn (Chốc lở, viêm mô tế bào...)",
    "Eczema Photos": "Bệnh chàm (Eczema)",
    "Exanthems and Drug Eruptions": "Phát ban và dị ứng do thuốc",
    "Herpes HPV and other STDs Photos": "Bệnh lây truyền qua đường tình dục (Herpes, HPV...)",
    "Light Diseases and Disorders of Pigmentation": "Rối loạn sắc tố và bệnh do ánh sáng",
    "Lupus and other Connective Tissue diseases": "Lupus và bệnh mô liên kết",
    "Melanoma Skin Cancer Nevi and Moles": "Ung thư hắc tố, nốt ruồi và bớt da",
    "Poison Ivy Photos and other Contact Dermatitis": "Viêm da tiếp xúc (Poison Ivy, dị ứng...)",
    "Psoriasis pictures Lichen Planus and related diseases": "Vảy nến và các bệnh da tương tự (Lichen Planus...)",
    "Seborrheic Keratoses and other Benign Tumors": "U lành tính và dày sừng tiết bã",
    "Systemic Disease": "Biểu hiện da của bệnh hệ thống",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Nhiễm nấm da (Hắc lào, Candida...)",
    "Urticaria Hives": "Mề đay (Urticaria)",
    "Vascular Tumors": "U mạch máu",
    "Vasculitis Photos": "Viêm mạch máu",
    "Warts Molluscum and other Viral Infections": "Mụn cóc, Molluscum và nhiễm virus khác"
}

# ==== 4. Hàm dự đoán ====
def predict_image(img_path):
    """Dự đoán 1 ảnh, trả về class và confidence"""
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        pred_class = class_names[pred_index]
        confidence = preds[0][pred_index] * 100
        return pred_class, confidence
    except Exception as e:
        raise RuntimeError(f"Lỗi khi dự đoán ảnh: {e}")

# ==== 5. Giao diện người dùng ====
class SkinDiagnosisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🩺 AI Chẩn đoán bệnh da")
        self.setGeometry(200, 200, 700, 500)

        # Label hiển thị ảnh
        self.image_label = QLabel("Chưa có ảnh")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid gray; text-align: center;")

        # Nút chọn ảnh
        self.btn_load = QPushButton("📂 Chọn ảnh")
        self.btn_load.clicked.connect(self.load_image)

        # Textbox nhập mô tả triệu chứng
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Nhập mô tả triệu chứng (ví dụ: ngứa, nổi mẩn, khô da...)")

        # Nút dự đoán
        self.btn_predict = QPushButton("🤖 Dự đoán")
        self.btn_predict.clicked.connect(self.make_prediction)

        # Label hiển thị kết quả
        self.result_label = QLabel("Kết quả dự đoán sẽ hiển thị ở đây")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-weight: bold; color: #333;")

        # Layout chính
        h_layout = QHBoxLayout()
        v_left = QVBoxLayout()
        v_left.addWidget(self.image_label)
        v_left.addWidget(self.btn_load)

        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("🧾 Mô tả triệu chứng:"))
        v_right.addWidget(self.desc_input)
        v_right.addWidget(self.btn_predict)
        v_right.addWidget(self.result_label)

        h_layout.addLayout(v_left)
        h_layout.addLayout(v_right)
        self.setLayout(h_layout)

        self.img_path = None

    def load_image(self):
        file_dialog = QFileDialog()
        fname, _ = file_dialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.img_path = fname
            pixmap = QPixmap(fname).scaled(self.image_label.width(), self.image_label.height())
            self.image_label.setPixmap(pixmap)

    def make_prediction(self):
        if not self.img_path:
            QMessageBox.warning(self, "Lỗi", "Bạn chưa chọn ảnh!")
            return
        try:
            pred_class, confidence = predict_image(self.img_path)
            pred_class_vi = label_map.get(pred_class, pred_class)  # Lấy tên tiếng Việt
            desc = self.desc_input.toPlainText().strip()
            advice = self.generate_advice(pred_class_vi, desc)
            #thêm tiếng việt
            pred_class_vi = label_map.get(pred_class, pred_class)
            self.result_label.setText(
                f"🧠 Dự đoán: <b>{pred_class_vi}</b> ({confidence:.2f}%)\n💡 Lời khuyên: {advice}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def generate_advice(self, pred_class_vi, desc):
        advice = f"Nếu tình trạng {pred_class_vi.lower()} kéo dài, bạn nên gặp bác sĩ da liễu."
        if "ngứa" in desc.lower():
            advice += " Có thể dùng thuốc bôi chống ngứa hoặc giữ da khô thoáng."
        if "mụn" in desc.lower():
            advice += " Nên giữ da sạch, tránh sờ tay lên vùng bị mụn."
        if "đỏ" in desc.lower():
            advice += " Hạn chế tiếp xúc ánh nắng và tránh sản phẩm gây kích ứng."
        return advice

# ==== 6. Chạy ứng dụng ====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinDiagnosisApp()
    window.show()
    sys.exit(app.exec_())
