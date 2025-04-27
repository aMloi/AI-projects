# Medicine Box OCR

## Overview

The Medicine Box OCR project is a full-stack web application that extracts medicine names from images of medicine boxes using Optical Character Recognition (OCR) and provides detailed information, including description, potential side effects, and generic alternatives. Built with Flask and EasyOCR, it uses free APIs (RxNorm and OpenFDA) to fetch drug data. The application features a user-friendly HTML interface for uploading images and displaying results. It is deployable using Docker or Heroku, making it suitable for healthcare or inventory management systems.

## Purpose

The purpose of this project is to:
- Automate the extraction of medicine names from medicine box images.
- Provide detailed medicine information using free APIs (RxNorm for descriptions and generics, OpenFDA for side effects).
- Demonstrate the integration of OCR, web development, and free API calls in a practical application.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload medicine box images via a web interface or POST request to the `/ocr` endpoint.
- Preprocess images with grayscale conversion, Gaussian blur, and Otsu's thresholding for improved OCR accuracy.
- Extract the most prominent text (medicine name) using EasyOCR.
- Fetch medicine details from free RxNorm (description, generics) and OpenFDA (side effects) APIs, with approximate matching for brand names.
- Display results in the web interface, including medicine description, up to 5 potential side effects with a disclaimer, and generic alternatives.
- Support for common image formats (e.g., JPG, PNG).
- Deployable as a Docker container or on Heroku.

## Project Structure

```
medicine-box-ocr/
├── Dockerfile          # Docker configuration for containerizing the app
├── Procfile            # Heroku process configuration
├── app.py              # Main Flask application with OCR and API integration
├── requirements.txt    # Python dependencies (e.g., Flask, EasyOCR, requests)
├── static/
│   └── style.css       # CSS for the web interface
├── templates/
│   └── index.html      # HTML frontend for image uploads and results
└── README.md           # This file
```

## Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git (for cloning the repository)
- Heroku CLI (optional, for Heroku deployment)
- Internet connection (for free API calls)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/medicine-box-ocr.git
   cd medicine-box-ocr
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   flask==2.0.1
   easyocr==1.7.1
   opencv-python==4.8.0.76
   numpy==1.24.3
   gunicorn==20.1.0
   requests==2.31.0
   ```
4. Ensure the `templates/` and `static/` folders are set up:
   - Place `index.html` in `templates/`.
   - Place `style.css` in `static/`.

*Note*: EasyOCR runs in CPU mode (`gpu=False`) for compatibility with free-tier environments. For GPU support, modify `app.py` and ensure a compatible setup.

## Usage

### Running Locally
1. Ensure dependencies are installed and the virtual environment is activated.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open a web browser and navigate to `http://localhost:5000`.
4. Use the web interface to:
   - Select and upload a medicine box image (JPG or PNG).
   - View the extracted medicine name, description, potential side effects (up to 5), and generic alternatives.
5. Alternatively, test the `/ocr` endpoint via `curl` or Postman:
   ```bash
   curl -X POST -F "image=@/path/to/medicine_box.jpg" http://localhost:5000/ocr
   ```
   Example response for "Augmentin":
   ```json
   {
       "name": "augmentin",
       "description": "Amoxicillin/clavulanate is a medication used for various conditions. Consult a healthcare provider for specific uses.",
       "side_effects": ["Renal failure acute", "Drug hypersensitivity", "Thrombocytopenia", "Petechiae", "Renal colic", "Note: This is not a complete list. Consult a healthcare provider."],
       "alternatives": ["amoxicillin/clavulanate"]
   }
   ```
   If no text is detected:
   ```json
   {"error": "No medicine name detected"}
   ```

### Testing with Sample Images
- Use clear, well-lit images of medicine boxes with prominent medicine names (e.g., "augmentin", "paracetamol", "ibuprofen").
- Ensure the name is readable for best OCR results.
- Test via the web interface or API with sample images.

### Error Handling
- Web interface: Displays errors like "No image uploaded" or "No medicine name detected."
- API: Returns JSON errors with HTTP 400 status:
  - `{"error": "No image provided"}`
  - `{"error": "Invalid image"}`
  - `{"error": "No medicine name detected"}`
- If API data is unavailable, fallback messages are shown (e.g., "No side effects data available").

## Deployment

### Deploying with Docker
1. Build the Docker image:
   ```bash
   docker build -t medicine-box-ocr .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 medicine-box-ocr
   ```
3. Access the app at `http://localhost:5000`.

### Deploying to Heroku
1. Log in to Heroku:
   ```bash
   heroku login
   ```
2. Create a Heroku app:
   ```bash
   heroku create [app-name]
   ```
3. Deploy the app:
   ```bash
   git push heroku main
   ```
4. Open the deployed app:
   ```bash
   heroku open
   ```
5. Access the web interface at `https://[app-name].herokuapp.com` or the API at `https://[app-name].herokuapp.com/ocr`.
6. Ensure `Procfile` contains:
   ```
   web: gunicorn app:app
   ```

*Note*: Heroku's free tier may have memory constraints for EasyOCR and API calls. Monitor performance and consider a paid dyno if needed. Ensure internet access for API requests.

## Troubleshooting

- **TemplateNotFound: index.html**:
  - Ensure `index.html` is in `templates/` in the same directory as `app.py`.
  - Verify the folder structure: `medicine_box_ocr/templates/index.html`.
  - Restart the Flask app after adding the file.
- **CSS Not Loading**:
  - Ensure `style.css` is in `medicine_box_ocr/static/`.
  - Check that `index.html` links to `/static/style.css`.
  - Test by accessing `http://localhost:5000/static/style.css` in a browser.
- **Slow OCR**:
  - EasyOCR in CPU mode can be slow. Use small images (e.g., 500x500 pixels) for testing.
  - Consider optimizing preprocessing or enabling GPU support if available.
- **API Errors**:
  - Ensure internet access for RxNorm and OpenFDA API calls.
  - Test with common US medicine names (e.g., "amoxicillin", "ibuprofen") for better API matches.
  - Check API rate limits and handle errors gracefully in `app.py`.
- **Limited API Data**:
  - For brand names like "Augmentin," ensure the image is clear. If data is missing, try the generic name (e.g., "amoxicillin/clavulanate").
- **Dependencies**:
  - Verify all dependencies are installed:
    ```bash
    pip show flask easyocr opencv-python numpy gunicorn requests
    ```
  - Reinstall if needed:
    ```bash
    pip install -r requirements.txt
    ```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/[your-feature-name]
   ```
3. Make your changes and commit them with clear messages:
   ```bash
   git commit -m "Add [description of changes]"
   ```
4. Push your changes to your fork:
   ```bash
   git push origin feature/[your-feature-name]
   ```
5. Submit a pull request with a detailed description of your changes.

Potential improvements:
- Enhance preprocessing for low-quality images.
- Add support for extracting additional fields (e.g., dosage, expiration date).
- Implement API response caching for faster results.
- Add unit tests for the API and frontend.

Please ensure your code follows the project's coding standards and includes documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
