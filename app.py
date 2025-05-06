from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load single model for now (used for all organs)
cae_model = tf.keras.models.load_model("cae_model.h5")

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def reconstruct_image(image_path):
    img = preprocess_image(image_path)
    reconstructed_img = cae_model.predict(img)[0]
    reconstructed_img = (reconstructed_img * 255).astype(np.uint8)
    reconstructed_img = reconstructed_img.squeeze()
    reconstructed_path = os.path.join(app.config["UPLOAD_FOLDER"], "reconstructed.png")
    cv2.imwrite(reconstructed_path, reconstructed_img)
    return reconstructed_path

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

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    denoised_path = reconstruct_image(filepath)

    return jsonify({
        "original": f"/static/uploads/{file.filename}",
        "denoised": f"/{denoised_path}"
    })

if __name__ == "__main__":
    app.run(debug=True)
