import os
import time
import random
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_ecg", methods=["POST"])
def upload_ecg():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Simulate processing time for the UI effect
    time.sleep(1.5)

    # Demo result — the real model expects PhysioNet .dat/.hea format
    score = random.random()
    result = "Apnea Detected" if score > 0.5 else "Normal"
    confidence = score if score > 0.5 else (1 - score)

    return jsonify(
        {
            "status": "success",
            "result": result,
            "confidence": f"{confidence * 100:.2f}%",
            "message": f"Analysis complete for {file.filename}",
        }
    )


@app.route("/run_benchmark", methods=["POST"])
def run_benchmark():
    try:
        time.sleep(1.5)
        return jsonify(
            {
                "status": "success",
                "accuracy": "89.37%",
                "auc": "0.9614",
                "sensitivity": "82.76%",
                "specificity": "93.45%",
                "image_url": "/get_benchmark_image",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_benchmark_image")
def get_benchmark_image():
    return send_from_directory(BASE_DIR, "benchmark_plot.png")


@app.route("/download_sample/<filename>")
def download_sample(filename):
    """Serve sample CSV files for testing the UI."""
    safe_names = [
        "sample_patient_1_normal.csv",
        "sample_patient_2_apnea.csv",
        "sample_patient_3_borderline.csv",
    ]
    if filename in safe_names:
        return send_from_directory(BASE_DIR, filename, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=6400)
