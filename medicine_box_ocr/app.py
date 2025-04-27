from flask import Flask, request, jsonify, render_template
import cv2
import easyocr
import numpy as np
import requests

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

def get_medicine_info(medicine_name):
    """Fetch medicine information from free RxNorm and OpenFDA APIs."""
    info = {
        "name": medicine_name,
        "description": "No description available. Consult a healthcare provider for details.",
        "side_effects": ["No side effects data available. Consult a healthcare provider."],
        "alternatives": ["No generic alternatives found. Consult a pharmacist."]
    }

    # Step 1: Use RxNorm for drug description and generics with approximate matching
    try:
        # Use approximate term search to handle brand names like "Augmentin"
        rxnorm_url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={medicine_name}&maxEntries=1"
        rxnorm_response = requests.get(rxnorm_url, timeout=5)
        rxnorm_response.raise_for_status()
        rxnorm_data = rxnorm_response.json()

        if "approximateGroup" in rxnorm_data and "candidate" in rxnorm_data["approximateGroup"]:
            rxcui = rxnorm_data["approximateGroup"]["candidate"][0]["rxcui"]
            # Get drug details
            details_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allProperties.json"
            details_response = requests.get(details_url, timeout=5)
            details_response.raise_for_status()
            details_data = details_response.json()

            for prop in details_data.get("propConceptGroup", {}).get("propConcept", []):
                if prop["propName"] == "RxNorm Name":
                    info["description"] = f"{prop['propValue']} is a medication used for various conditions. Consult a healthcare provider for specific uses."

            # Get generic alternatives
            generic_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?relas=tradename_of"
            generic_response = requests.get(generic_url, timeout=5)
            generic_response.raise_for_status()
            generic_data = generic_response.json()
            info["alternatives"] = []
            for concept in generic_data.get("relatedGroup", {}).get("conceptGroup", []):
                if concept.get("tty") in ["IN", "MIN"]:  # Ingredient or Multi-Ingredient
                    for c in concept.get("conceptProperties", []):
                        info["alternatives"].append(c["name"])
            if not info["alternatives"]:
                info["alternatives"] = ["No generic alternatives found. Consult a pharmacist."]
    except requests.RequestException:
        pass  # Retain default info values

    # Step 2: Use OpenFDA for side effects, limited to top 5
    try:
        openfda_url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{medicine_name}&limit=10"
        openfda_response = requests.get(openfda_url, timeout=5)
        openfda_response.raise_for_status()
        openfda_data = openfda_response.json()

        if "results" in openfda_data:
            info["side_effects"] = []
            for event in openfda_data["results"]:
                for reaction in event.get("patient", {}).get("reaction", []):
                    side_effect = reaction.get("reactionmeddrapt", "")
                    if side_effect and side_effect not in info["side_effects"]:
                        info["side_effects"].append(side_effect)
            # Limit to top 5 side effects for readability
            info["side_effects"] = info["side_effects"][:5]
            if not info["side_effects"]:
                info["side_effects"] = ["No side effects data available. Consult a healthcare provider."]
            else:
                info["side_effects"].append("Note: This is not a complete list. Consult a healthcare provider.")
    except requests.RequestException:
        pass  # Retain default side effects value

    return info

@app.route('/', methods=['GET'])
def index():
    """Serve the HTML frontend."""
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    """Handle image upload and return extracted medicine information."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    thresh = preprocess_image(image)
    medicine_name = extract_medicine_name(thresh)
    if not medicine_name:
        return jsonify({"error": "No medicine name detected"}), 400
    medicine_info = get_medicine_info(medicine_name)
    return jsonify(medicine_info)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
