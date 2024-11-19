from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Inisialisasi Flask
app = Flask(__name__)

# Load model
MODEL_PATH = "Finalmodel_keras2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocessing fungsi
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preproses gambar sebelum digunakan untuk prediksi.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk menerima file gambar dan melakukan prediksi.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Simpan file sementara
        temp_path = "temp_image.jpg"
        file.save(temp_path)

        # Preproses gambar
        img_array = preprocess_image(temp_path)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
