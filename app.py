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
from werkzeug.utils import secure_filename

# ===== Cấu hình =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "cnn_model_final.h5"
IMG_SIZE = (192, 192)
SYMPTOM_FILE = "skin_diseases.xlsx"
ALPHA = 0.7  # Tỷ lệ kết hợp giữa ảnh và mô tả

# ===== Tải mô hình =====
model = load_model(MODEL_PATH)

# ===== Danh sách nhãn bệnh =====
class_names = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions",
    "Atopic Dermatitis",
    "Cellulitis, Impetigo, and other Bacterial Infections",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Herpes, HPV, and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue Diseases",
    "Melanoma, Skin Cancer, Nevi, and Moles",
    "Poison Ivy and Contact Dermatitis",
    "Psoriasis, Lichen Planus, and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea, Ringworm, Candidiasis, and other Fungal Infections",
    "Urticaria",
    "Vascular Tumors",
    "Vasculitis",
    "Warts, Molluscum, and other Viral Infections"
]

# ===== Bản dịch tiếng Việt =====
label_map = {
    "Acne and Rosacea": "Mụn trứng cá và hoặc các dạng mụn nói chung",
    "Actinic Keratosis, Basal Cell Carcinoma, and other Malignant Lesions": "Dày sừng ánh sáng, ung thư biểu mô tế bào đáy và các tổn thương ác tính khác",
    "Atopic Dermatitis": "Viêm da cơ địa",
    "Cellulitis, Impetigo, and other Bacterial Infections": "Viêm mô tế bào, chốc lở và các nhiễm trùng da do vi khuẩn khác",
    "Eczema": "Chàm (Eczema)",
    "Exanthems and Drug Eruptions": "Phát ban do virus hoặc phản ứng thuốc",
    "Herpes, HPV, and other STDs": "Mụn rộp, HPV và các bệnh lây qua đường tình dục khác",
    "Light Diseases and Disorders of Pigmentation": "Rối loạn sắc tố và bệnh lý do ánh sáng",
    "Lupus and other Connective Tissue Diseases": "Lupus và các bệnh mô liên kết khác",
    "Melanoma, Skin Cancer, Nevi, and Moles": "U hắc tố, ung thư da, nốt ruồi và bớt",
    "Poison Ivy and Contact Dermatitis": "Viêm da tiếp xúc (do cây thường xuân hoặc chất gây kích ứng)",
    "Psoriasis, Lichen Planus, and related diseases": "Vảy nến, lichen phẳng và các bệnh liên quan",
    "Seborrheic Keratoses and other Benign Tumors": "Dày sừng tiết bã và các khối u lành tính khác",
    "Systemic Disease": "Bệnh lý hệ thống",
    "Tinea, Ringworm, Candidiasis, and other Fungal Infections": "Nấm da, hắc lào, candida và các nhiễm nấm khác",
    "Urticaria": "Mề đay",
    "Vascular Tumors": "U mạch máu",
    "Vasculitis": "Viêm mạch máu",
    "Warts, Molluscum, and other Viral Infections": "Mụn cóc, u mềm lây và các nhiễm virus khác"
}

# ===== Dữ liệu triệu chứng =====
df_symptoms = pd.read_excel(SYMPTOM_FILE)
if 'Disease' not in df_symptoms.columns or 'Symptom' not in df_symptoms.columns:
    raise ValueError("File triệu chứng cần có cột 'Disease' và 'Symptom'.")

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
