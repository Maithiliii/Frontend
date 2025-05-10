from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Load models
brain_autoencoder = tf.keras.models.load_model("autoencoder3_brain_model.h5", compile=False)
kidney_autoencoder = tf.keras.models.load_model("autoencoder4_kidney_model.h5", compile=False)

brain_classifier = tf.keras.models.load_model("best_model.h5", compile=False)
kidney_classifier = tf.keras.models.load_model("tumor_classifier3_kidney.h5", compile=False)

brain_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
kidney_labels = ['no tumor', 'tumor']

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def reconstruct_image(image_path, organ, filename):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256)).astype("float32") / 255.0
    input_img = np.expand_dims(img, axis=0)

    if organ == "brain":
        output_img = brain_autoencoder.predict(input_img)[0]
    elif organ == "kidney":
        output_img = kidney_autoencoder.predict(input_img)[0]
    else:
        raise ValueError("Unsupported organ")

    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    recon_filename = f"reconstructed_{organ}_{filename}"
    recon_path = os.path.join(PROCESSED_FOLDER, recon_filename)
    cv2.imwrite(recon_path, output_img)

    return recon_filename

def classify_image(image_path, organ):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (240, 240)).astype("float32") / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    if organ == "brain":
        preds = brain_classifier.predict(img)
        return brain_labels[np.argmax(preds)]
    elif organ == "kidney":
        preds = kidney_classifier.predict(img)
        return kidney_labels[int(np.round(preds[0][0]))]
    else:
        return "Invalid organ"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload/<organ>")
def upload_page(organ):
    return render_template("upload.html", organ=organ)

@app.route("/process/<organ>", methods=["POST"])
def process(organ):
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        recon_filename = reconstruct_image(filepath, organ, filename)
        prediction = classify_image(filepath, organ)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Ensure we return the static paths for the images
    return jsonify({
        "original": f"/static/uploads/{filename}",
        "denoised": f"/static/processed/{recon_filename}",
        "prediction": prediction
    })

# Flask route to serve static files (if needed)
@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
