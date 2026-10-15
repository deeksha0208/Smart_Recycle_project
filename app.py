
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline
import io

app = Flask(__name__)
CORS(app)

# Load the waste-classification AI model
print("Loading AI model. Please wait...")
classifier = pipeline(
    "image-classification",
    model="watersplash/waste-classification"
)
print("AI model loaded successfully!")

tips = {
    "plastic": "Clean the item and check whether your local recycling service accepts it.",
    "paper": "Keep paper clean and dry before recycling.",
    "cardboard": "Flatten cardboard boxes and keep them dry.",
    "metal": "Rinse cans and containers before recycling.",
    "green-glass": "Separate glass carefully and follow local glass-recycling rules.",
    "white-glass": "Separate glass carefully and follow local glass-recycling rules.",
    "biological": "Place suitable food and garden waste in a compost or organic-waste bin.",
    "battery": "Take batteries to an authorised battery collection point.",
    "clothes": "Donate wearable clothes or look for a textile-recycling service.",
    "shoes": "Donate usable shoes or check for a textile-recycling collection point.",
    "trash": "Check local disposal rules before placing this item in general waste."
}

@app.route("/")
def home():
    return "ReCycle AI backend is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        results = classifier(image)

        best = results[0]
        category = best["label"]
        confidence = round(float(best["score"]) * 100, 2)

        return jsonify({
            "category": category,
            "confidence": confidence,
            "tip": tips.get(category.lower(), "Follow your local waste-disposal guidelines.")
        })

    except Exception as error:
        print("Prediction error:", error)
        return jsonify({"error": "Could not analyse this image."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
