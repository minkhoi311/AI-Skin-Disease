import sys
import os
import numpy as np
from googletrans import Translator
import pandas as pd
import tensorflow as tf
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== Cấu hình =====
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)

MODEL_PATH = "cnn_model_final.h5"
IMG_SIZE = (192, 192)
DATASET_PATH = "Dataset/train"
SYMPTOM_FILE = "skin_diseases.xlsx"
ALPHA = 0.7  # Ảnh chiếm 70%, text chiếm 30%

# ===== Load lớp =====
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {DATASET_PATH}")
class_names = sorted(os.listdir(DATASET_PATH))

# ===== Load model CNN =====
model = load_model(MODEL_PATH)


# Hàm dịch tự động
def auto_translate_labels(class_list):
    translator = Translator()
    label_map = {}
    for name in class_list:
        vi_name = translator.translate(name, src='en', dest='vi').text
        label_map[name] = vi_name
    return label_map

# Dịch sang tiếng Việt
label_map = auto_translate_labels(class_names)

# ===== Load triệu chứng =====
df_symptoms = pd.read_excel(SYMPTOM_FILE)
if 'Disease' not in df_symptoms.columns or 'Symptom' not in df_symptoms.columns:
    raise ValueError("File triệu chứng cần có cột 'Disease' và 'Symptom'.")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_symptoms['Symptom'])


# ==== Hàm tiền xử lý text ====
def preprocess_text(text):
    """Chuẩn hóa, mở rộng ý nghĩa từ mô tả của người dùng."""
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', '', text)
    # mở rộng một số từ khóa phổ biến
    text = text.replace("mụn", "mụn trứng cá đỏ mụn nhỏ trên mặt")
    text = text.replace("ngứa", "ngứa khô rát khó chịu")
    text = text.replace("rát", "rát đỏ sưng")
    text = text.replace("bong", "bong tróc tróc vảy")
    text = text.replace("vảy", "vảy trắng da khô")
    return text.strip()


# ==== Dự đoán ảnh ====
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]


# ==== Dự đoán triệu chứng ====
def predict_from_text(patient_desc):
    if not patient_desc.strip():
        return np.zeros(len(class_names))

    desc = preprocess_text(patient_desc)
    vec = vectorizer.transform([desc])
    sims = cosine_similarity(vec, tfidf_matrix)[0]

    text_scores = np.zeros(len(class_names))
    for i, disease in enumerate(class_names):
        # khớp gần đúng
        matches = [
            j for j, d in enumerate(df_symptoms['Disease'])
            if d.lower() in disease.lower() or disease.lower() in d.lower()
        ]
        if matches:
            text_scores[i] = np.max(sims[matches])
        else:
            text_scores[i] = 0
    return text_scores


# ===== Giao diện GUI =====
class SkinDiagnosisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🩺 AI Chẩn đoán bệnh da")
        self.setGeometry(200, 200, 900, 520)

        self.image_label = QLabel("Chưa có ảnh")
        self.image_label.setFixedSize(224, 224)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.btn_load = QPushButton("📂 Chọn ảnh")
        self.btn_load.clicked.connect(self.load_image)

        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Nhập mô tả triệu chứng (ví dụ: ngứa, nổi mụn, khô da...)")
        self.desc_input.setFixedHeight(200)

        self.btn_predict = QPushButton("🤖 Dự đoán")
        self.btn_predict.clicked.connect(self.make_prediction)

        self.result_label = QLabel("Kết quả dự đoán sẽ hiển thị ở đây")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-weight: bold; color: #333;")

        v_left = QVBoxLayout()
        v_left.addWidget(self.image_label)
        v_left.addWidget(self.btn_load)

        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("🧾 Mô tả triệu chứng:"))
        v_right.addWidget(self.desc_input)
        v_right.addWidget(self.btn_predict)
        v_right.addWidget(self.result_label)

        h_layout = QHBoxLayout()
        h_layout.addLayout(v_left)
        h_layout.addLayout(v_right)
        self.setLayout(h_layout)

        self.img_path = None

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.img_path = fname
            pixmap = QPixmap(fname).scaled(self.image_label.width(), self.image_label.height())
            self.image_label.setPixmap(pixmap)

    def make_prediction(self):
        if not self.img_path:
            QMessageBox.warning(self, "Lỗi", "Bạn chưa chọn ảnh!")
            return
        try:
            img_probs = predict_image(self.img_path)
            desc = self.desc_input.toPlainText().strip()
            text_scores = predict_from_text(desc)

            final_scores = ALPHA * img_probs + (1 - ALPHA) * text_scores
            final_index = np.argmax(final_scores)
            final_class = class_names[final_index]
            final_class_vi = label_map.get(final_class, final_class)

            advice = self.generate_advice(final_class_vi, desc)
            self.result_label.setText(
                f"🧠 <b>Kết quả:</b> {final_class_vi}\n💡 {advice}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def generate_advice(self, pred_class_vi, desc):
        advice = f"Nếu tình trạng {pred_class_vi.lower()} kéo dài, bạn nên gặp bác sĩ da liễu."
        if desc:
            desc_lower = desc.lower()
            if "ngứa" in desc_lower:
                advice += " Có thể dùng thuốc bôi chống ngứa hoặc giữ da khô thoáng."
            if "mụn" in desc_lower:
                advice += " Nên giữ da sạch, tránh sờ tay lên vùng bị mụn."
            if "đỏ" in desc_lower:
                advice += " Hạn chế tiếp xúc ánh nắng và tránh sản phẩm gây kích ứng."
            if "tróc" in desc_lower or "bong" in desc_lower:
                advice += " Nên dưỡng ẩm thường xuyên để tránh khô da."
        return advice


# ===== Run App =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinDiagnosisApp()
    window.show()
    sys.exit(app.exec_())
