import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys

# ==== 1. Cấu hình ====
MODEL_PATH = "test.h5"  # ⚠️ thay bằng file .h5 của bạn
DATASET_PATH = "Dataset/train"  # để lấy tên lớp
IMG_SIZE = (192, 192)

# ==== 2. Lấy danh sách lớp tự động ====
try:
    class_names = sorted([d for d in os.listdir(DATASET_PATH)
                          if os.path.isdir(os.path.join(DATASET_PATH, d))])
    num_classes = len(class_names)
    print(f"🔍 Tìm thấy {num_classes} lớp:")
    for i, name in enumerate(class_names):
        print(f"  {i + 1}. {name}")
except Exception as e:
    print(f"❌ Lỗi khi đọc thư mục dataset: {e}")
    sys.exit(1)

# ==== 3. Nạp mô hình đã train ====
try:
    print(f"\n📂 Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.\n")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    sys.exit(1)


# ==== 4. Hàm dự đoán cho 1 ảnh - CẢI TIẾN ====
def predict_image(img_path, show_image=True, top_k=3):
    """Dự đoán bệnh da liễu từ ảnh"""
    if not os.path.exists(img_path):
        print(f"⚠️ File not found: {img_path}")
        return None

    try:
        # Kiểm tra định dạng file
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        if not img_path.lower().endswith(valid_ext):
            print(f"⚠️ Định dạng không hỗ trợ: {img_path}")
            return None

        # Load ảnh và tiền xử lý
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        preds = model.predict(img_array, verbose=0)

        # Lấy top k kết quả
        top_indices = np.argsort(preds[0])[-top_k:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_confidences = [preds[0][i] * 100 for i in top_indices]

        # Kết quả chính
        pred_class = top_classes[0]
        confidence = top_confidences[0]

        if show_image:
            plt.figure(figsize=(10, 5))

            # Hiển thị ảnh
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Dự đoán: {pred_class}\nĐộ tin cậy: {confidence:.2f}%")
            plt.axis('off')

            # Hiển thị biểu đồ xác suất
            plt.subplot(1, 2, 2)
            y_pos = np.arange(len(top_classes))
            plt.barh(y_pos, top_confidences, color='skyblue')
            plt.yticks(y_pos, top_classes)
            plt.xlabel('Độ tin cậy (%)')
            plt.title('Top {} dự đoán'.format(top_k))
            plt.gca().invert_yaxis()

            plt.tight_layout()
            plt.show()

        print(f"🎯 Kết quả cho: {os.path.basename(img_path)}")
        print(f"🏥 Dự đoán chính: {pred_class} ({confidence:.2f}%)")
        print("📊 Top {} dự đoán:".format(top_k))
        for i, (cls, conf) in enumerate(zip(top_classes, top_confidences)):
            print(f"   {i + 1}. {cls}: {conf:.2f}%")
        print()

        return pred_class, confidence, list(zip(top_classes, top_confidences))

    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh {img_path}: {e}")
        return None


# ==== 5. Dự đoán nhiều ảnh trong 1 thư mục - CẢI TIẾN ====
def batch_predict(folder_path, show_images=False, max_images=None):
    """Dự đoán cho tất cả ảnh trong thư mục"""
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return []

    # Tìm file ảnh
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(valid_ext)]

    if not files:
        print("❌ Không tìm thấy file ảnh nào trong thư mục")
        return []

    if max_images:
        files = files[:max_images]

    print(f"📁 Đang xử lý {len(files)} ảnh từ: {folder_path}")
    results = []

    for i, fname in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Đang xử lý: {fname}")
        full_path = os.path.join(folder_path, fname)

        res = predict_image(full_path, show_image=show_images)
        if res:
            pred_class, confidence, all_preds = res
            results.append({
                'filename': fname,
                'prediction': pred_class,
                'confidence': confidence,
                'all_predictions': all_preds
            })

    # Tóm tắt kết quả
    if results:
        print("\n" + "=" * 50)
        print("📈 TÓM TẮT KẾT QUẢ")
        print("=" * 50)
        for result in results:
            print(f"📄 {result['filename']} → {result['prediction']} ({result['confidence']:.2f}%)")

    return results


# ==== 6. Chạy từ terminal - CẢI TIẾN ====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
🤖 SKIN DISEASE PREDICTION TOOL

Cách sử dụng:
  python predict_skin_diseases.py <path_to_image_or_folder> [max_images]

Ví dụ:
  python predict_skin_diseases.py test_image.jpg
  python predict_skin_diseases.py test_folder 5
        """)
        sys.exit(1)

    path = sys.argv[1]
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("🩺 BẮT ĐẦU DỰ ĐOÁN BỆNH DA LIỄU")
    print("=" * 40)

    if os.path.isdir(path):
        results = batch_predict(path, show_images=True, max_images=max_images)
    else:
        result = predict_image(path, show_image=True)
        if result:
            pred_class, confidence, all_preds = result