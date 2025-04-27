# Medicine Box OCR

## Overview

The Medicine Box OCR project is a web application that extracts medicine names from images of medicine boxes using Optical Character Recognition (OCR). Built with Flask and EasyOCR, the application processes uploaded images through a preprocessing pipeline and returns the most prominent text (assumed to be the medicine name) in JSON format. The project is designed for easy deployment using Docker or Heroku, making it suitable for integration into healthcare or inventory management systems.

## Purpose

The purpose of this project is to:
- Automate the extraction of medicine names from images of medicine boxes.
- Provide a simple API endpoint for OCR functionality.
- Demonstrate the use of EasyOCR with image preprocessing for text extraction in a real-world application.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload images of medicine boxes via a POST request to the `/ocr` endpoint.
- Preprocess images with grayscale conversion, Gaussian blur, and Otsu's thresholding for improved OCR accuracy.
- Extract the most prominent text (medicine name) using EasyOCR, prioritizing text with the largest height.
- Return extracted medicine name in JSON format (lowercase, cleaned of whitespace).
- Support for common image formats (e.g., JPG, PNG).
- Deployable as a Docker container or on Heroku.

## Project Structure

```
medicine-box-ocr/
├── Dockerfile          # Docker configuration for containerizing the app
├── Procfile            # Heroku process configuration
├── app.py              # Main Flask application with OCR logic
├── requirements.txt    # Python dependencies (e.g., Flask, EasyOCR, OpenCV)
└── README.md           # This file
```

## Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git (for cloning the repository)
- Heroku CLI (optional, for Heroku deployment)

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
   Ensure `requirements.txt` includes:
   ```
   flask==2.0.1
   easyocr==1.7.1
   opencv-python==4.8.0.76
   numpy==1.24.3
   gunicorn==20.1.0  # For Heroku deployment
   ```

*Note*: EasyOCR uses CPU mode by default in this project (`gpu=False`) to support free-tier deployment environments. For GPU support, update `app.py` and ensure a compatible environment.

## Usage

### Running Locally
1. Ensure dependencies are installed and the virtual environment is activated.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. The app will start on `http://localhost:5000`.
4. Test the OCR endpoint using a tool like `curl` or Postman:
   ```bash
   curl -X POST -F "image=@/path/to/medicine_box.jpg" http://localhost:5000/ocr
   ```
5. Example response:
   ```json
   {"medicine_name": "paracetamol"}
   ```
   If no valid text is detected, the response will be:
   ```json
   {"medicine_name": ""}
   ```

### Testing with Sample Images
- Use clear, well-lit images of medicine boxes in JPG or PNG format.
- Ensure the medicine name is prominent and readable for best results.
- Test with a sample image via `curl` or a custom HTML form (not included in this project but can be added).

### Error Handling
- If no image is provided: `{"error": "No image provided"}` (HTTP 400)
- If the image is invalid: `{"error": "Invalid image"}` (HTTP 400)

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
3. Access the app at `http://localhost:5000/ocr`.

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
5. Test the `/ocr` endpoint using the Heroku app URL (e.g., `https://[app-name].herokuapp.com/ocr`).
6. Ensure `Procfile` contains:
   ```
   web: gunicorn app:app
   ```

*Note*: Heroku's free tier may have memory limitations for EasyOCR. Monitor performance and consider a paid dyno if needed.

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
- Add a web interface for image uploads.
- Enhance preprocessing for better OCR accuracy.
- Support additional languages in EasyOCR.
- Optimize for GPU environments.

Please ensure your code follows the project's coding standards and includes documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
