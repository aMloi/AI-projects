# Medicine Box OCR

## Overview

The Medicine Box OCR project is a web application designed to extract text from images of medicine boxes using Optical Character Recognition (OCR). This tool helps users quickly identify key information such as drug names, dosages, and expiration dates from medicine packaging. The application is built with Python, leverages OCR libraries, and is deployable as a web service using Docker and a platform like Heroku.

## Purpose

The purpose of this project is to:
- Automate the extraction of text from medicine box images.
- Provide a user-friendly web interface for uploading images and viewing extracted text.
- Demonstrate the application of OCR in healthcare-related use cases.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload images of medicine boxes via a web interface.
- Extract text using OCR (e.g., drug names, dosages, expiration dates).
- Display extracted text in a clean, readable format.
- Support for common image formats (e.g., JPG, PNG).
- Deployable as a Docker container or on platforms like Heroku.

## Project Structure

```
medicine-box-ocr/
├── Dockerfile          # Docker configuration for containerizing the app
├── Procfile            # Heroku process configuration
├── app.py              # Main Flask application script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Docker (optional, for containerized deployment)
- Git (for cloning the repository)
- Tesseract OCR (if used; requires system installation)
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
4. If using Tesseract OCR, install it on your system:
   - On Ubuntu:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - On macOS:
     ```bash
     brew install tesseract
     ```
   - On Windows: Download and install from [Tesseract's GitHub](https://github.com/UB-Mannheim/tesseract/wiki).
5. Verify Tesseract installation:
   ```bash
   tesseract --version
   ```

## Usage

### Running Locally
1. Ensure dependencies are installed and the virtual environment is activated.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open a web browser and navigate to `http://localhost:5000` (or the port specified in `app.py`).
4. Upload a medicine box image using the web interface.
5. View the extracted text displayed on the page.

### Testing with Sample Images
- Place sample medicine box images (e.g., JPG or PNG) in a local directory.
- Use the web interface to upload images and test the OCR functionality.
- Ensure images are clear and well-lit for optimal OCR accuracy.

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
5. Ensure Heroku-specific dependencies (e.g., `gunicorn`) are included in `requirements.txt`.

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

Please ensure your code follows the project's coding standards and includes documentation where necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
