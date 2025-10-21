import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from werkzeug.utils import secure_filename

# ===== Cấu hình =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "cnn_model_final.h5"
IMG_SIZE = (192, 192)
DATASET_PATH = "Dataset/train"
SYMPTOM_FILE = "skin_diseases.xlsx"
ALPHA = 0.7

# ===== Tải mô hình và dữ liệu =====
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {DATASET_PATH}")
class_names = sorted(os.listdir(DATASET_PATH))
model = load_model(MODEL_PATH)


# ===== Dịch nhãn sang tiếng Việt =====
def auto_translate_labels(class_list):
    translator = Translator()
    label_map = {}
    for name in class_list:
        try:
            vi_name = translator.translate(name, src='en', dest='vi').text
        except Exception:
            vi_name = name
        label_map[name] = vi_name
    return label_map


label_map = auto_translate_labels(class_names)

# ===== Dữ liệu triệu chứng =====
df_symptoms = pd.read_excel(SYMPTOM_FILE)
if 'Disease' not in df_symptoms.columns or 'Symptom' not in df_symptoms.columns:
    raise ValueError("File triệu chứng cần có cột 'Disease' và 'Symptom'.")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_symptoms['Symptom'])


# ===== Hàm xử lý =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', '', text)
    text = text.replace("mụn đỏ", "mụn trứng cá đỏ mụn nhỏ trên mặt")
    text = text.replace("ngứa", "ngứa khô rát khó chịu")
    text = text.replace("rát", "rát đỏ sưng")
    text = text.replace("bong", "bong tróc tróc vảy")
    text = text.replace("vảy", "vảy trắng da khô")
    return text.strip()


def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return preds[0]


def predict_from_text(patient_desc):
    if not patient_desc.strip():
        return np.zeros(len(class_names))
    desc = preprocess_text(patient_desc)
    vec = vectorizer.transform([desc])
    sims = cosine_similarity(vec, tfidf_matrix)[0]
    text_scores = np.zeros(len(class_names))
    for i, disease in enumerate(class_names):
        matches = [
            j for j, d in enumerate(df_symptoms['Disease'])
            if d.lower() in disease.lower() or disease.lower() in d.lower()
        ]
        if matches:
            text_scores[i] = np.max(sims[matches])
    return text_scores


def generate_advice(pred_class_vi, desc):
    advice = f"Nếu tình trạng {pred_class_vi.lower()} kéo dài, hãy gặp bác sĩ da liễu sớm."
    if desc:
        desc_lower = desc.lower()
        if "ngứa" in desc_lower:
            advice += " Bạn nên dùng kem làm dịu và tránh gãi nhiều."
        if "mụn" in desc_lower:
            advice += " Rửa mặt nhẹ, tránh mỹ phẩm gây bít lỗ chân lông."
        if "đỏ" in desc_lower:
            advice += " Hạn chế nắng, tránh nhiệt độ cao."
        if "bong" in desc_lower or "tróc" in desc_lower:
            advice += " Nên dưỡng ẩm thường xuyên."
    return advice


# ===== Flask Routes =====
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    advice = None
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        desc = request.form.get('desc', '')

        if not file or file.filename == '':
            result = "Vui lòng tải ảnh lên."
        else:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            img_probs = predict_image(img_path)
            text_scores = predict_from_text(desc)
            final_scores = ALPHA * img_probs + (1 - ALPHA) * text_scores
            final_index = np.argmax(final_scores)
            final_class = class_names[final_index]
            final_class_vi = label_map.get(final_class, final_class)
            advice = generate_advice(final_class_vi, desc)
            result = f"Dự đoán: {final_class_vi}"

    return render_template('index.html', result=result, advice=advice, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
