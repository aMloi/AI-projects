from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np

app = Flask(__name__)

def preprocess_image(image):
    """Read and preprocess the image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_medicine_name(thresh):
    """Extract medicine name using EasyOCR, based on largest text height."""
    reader = easyocr.Reader(['en'], gpu=False)  # CPU for free tier
    results = reader.readtext(thresh)
    if not results:
        return ""
    results_sorted = sorted(results, key=lambda r: abs(r[0][0][1] - r[0][3][1]), reverse=True)
    for bbox, text, conf in results_sorted:
        cleaned = text.strip()
        if len(cleaned) > 2 and any(c.isalpha() for c in cleaned):
            return cleaned.lower()
    return ""

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    thresh = preprocess_image(image)
    medicine_name = extract_medicine_name(thresh)
    return jsonify({"medicine_name": medicine_name})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)