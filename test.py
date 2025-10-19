import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ==== 1. Cấu hình ====
MODEL_PATH = "best_model.h5"  # ⚠️ thay bằng file .h5 của bạn
DATASET_PATH = "Dataset/train"  # để lấy tên lớp
IMG_SIZE = (192, 192)

# ==== 2. Lấy danh sách lớp tự động ====
class_names = sorted(os.listdir(DATASET_PATH))
num_classes = len(class_names)
print(f"🔍 Tìm thấy {num_classes} lớp:")
print(class_names)

# ==== 3. Nạp mô hình đã train ====
print(f"\n📂 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.\n")

# ==== 4. Hàm dự đoán cho 1 ảnh ====
def predict_image(img_path, show_image=True):
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}")
        return None

    # Load ảnh và tiền xử lý
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # In thông tin và hình ảnh sau khi xử lý
    plt.figure(figsize=(4, 4))
    plt.imshow(np.array(img).astype(np.uint8)) # chuyển ngược về dạng hiển thị được
    plt.title(f"Hình đã resize: {IMG_SIZE}")
    plt.axis("off")
    plt.show()

    print(f"Shape sau tiền xử lý: {img_array.shape}")
    print(f"Giá trị pixel [0,0,:]: {img_array[0, 0, 0, :]}")
    # Dự đoán
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index] * 100

    if show_image:
        plt.imshow(img)
        plt.title(f"Predicted: {pred_class} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()

    return pred_class, confidence

# ==== 5. Dự đoán nhiều ảnh trong 1 thư mục ====
def batch_predict(folder_path, show_images=False, max_images=None):
    results = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for i, fname in enumerate(files):
        if max_images and i >= max_images:
            break
        full_path = os.path.join(folder_path, fname)
        res = predict_image(full_path, show_image=show_images)
        if res:
            results.append((fname, res[0], res[1]))
    return results

# ==== 6. Chạy từ terminal ====
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_skin_diseases.py <path_to_image_or_folder>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        print(f"📁 Dự đoán cho các ảnh trong thư mục: {path}")
        preds = batch_predict(path, show_images=True, max_images=10)
        for fname, label, conf in preds:
            print(f"{fname} → {label} ({conf:.2f}%)")
    else:
        label, conf = predict_image(path, show_image=True)
        print(f"🩺 Dự đoán: {label} ({conf:.2f}%)")
