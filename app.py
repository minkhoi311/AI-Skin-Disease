import os
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename


# ===== Khởi tạo mô hình & dữ liệu =====
global model, class_names, label_map, vectorizer, tfidf_matrix, df_symptoms

MODEL_PATH = "cnn_model_final.h5"
DISEASE_FILE = "data/diseases.json"
IMG_SIZE = (192, 192)
SYMPTOM_FILE = "skin_diseases.xlsx"
ALPHA = 0.7  # Tỷ lệ kết hợp giữa ảnh và mô tả


# ===== Tải mô hình =====
model = load_model(MODEL_PATH)


#Load class_names & label_map từ JSON
with open(DISEASE_FILE, "r", encoding="utf-8") as f:
    disease_data = json.load(f)

class_names = disease_data["class_names"]
label_map = disease_data["label_map"]

# ===== Dữ liệu triệu chứng =====
df_symptoms = pd.read_excel(SYMPTOM_FILE)
if 'Disease' not in df_symptoms.columns or 'Symptom' not in df_symptoms.columns:
    raise ValueError("File triệu chứng cần có cột 'Disease' và 'Symptom'.")


# ===== Cấu hình =====
def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app


app = create_app()

df_symptoms = pd.read_excel("skin_diseases.xlsx")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_symptoms['Symptom'])


# ===== Các hàm xử lý =====
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

