from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

cae_model = tf.keras.models.load_model("brainkidney.h5", compile=False)
classifier_model = tf.keras.models.load_model("best_model.h5", compile=False)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def reconstruct_image(image_path):
    img = cv2.imread(image_path)

    img = cv2.resize(img, (256, 256)).astype("float32") / 255.0  

    input_img = np.expand_dims(img, axis=0)

    output_img = cae_model.predict(input_img)[0]

    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    reconstructed_path = os.path.join(app.config["UPLOAD_FOLDER"], "reconstructed.png")
    cv2.imwrite(reconstructed_path, output_img)

    return reconstructed_path

def classify_tumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (240, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = classifier_model.predict(img)
    return class_labels[np.argmax(pred)]

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
    
    predicted_class = classify_tumor(filepath)

    return jsonify({
        "original": f"/static/uploads/{file.filename}",
        "denoised": f"/static/uploads/reconstructed.png",  
        "prediction": predicted_class
    })

if __name__ == "__main__":
    app.run(debug=True)
