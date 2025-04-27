# Waste Classification Using Transfer Learning

## Project Overview

This project automates the classification of waste products into recyclable and organic categories using machine learning and computer vision. By leveraging transfer learning with a pre-trained VGG16 model, the project addresses the inefficiencies of manual waste sorting, reducing labor costs and contamination rates for EcoClean.

## Aim of the Project

The goal is to develop a robust model that accurately distinguishes between recyclable and organic waste based on image data. The final output is a trained model that classifies waste images into these two categories with high accuracy.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project environment, install the required dependencies using pip:

```bash
pip install tensorflow==2.17.0
pip install numpy==1.24.3
pip install scikit-learn==1.5.1
pip install matplotlib==3.9.2
```

Ensure you have Python 3.8 or higher installed. The project was developed and tested in a Google Colab environment.

## Dataset

The dataset used is `o-vs-r-split-reduced-1200.zip`, containing images of organic (O) and recyclable (R) waste, split into training and testing sets. The dataset is downloaded and extracted automatically during script execution from the following URL:

```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip
```

- Training Set: Used for model training and validation (80% training, 20% validation).
- Testing Set: Used for final model evaluation.
- Image Size: Resized to 150x150 pixels.
- Classes: Organic (O), Recyclable (R).

## Project Structure

```
waste-classification/
├── o-vs-r-split/
│   ├── train/
│   │   ├── O/
│   │   ├── R/
│   ├── test/
│   │   ├── O/
│   │   ├── R/
├── O_R_tlearn_vgg16.keras
├── O_R_tlearn_fine_tune_vgg16.keras
├── Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb
└── README.md
```

## Usage

1. Clone the repository or download the project files.
2. Open the Jupyter notebook `Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb` in Google Colab or a local Jupyter environment.
3. Run the notebook cells sequentially to:
   - Install dependencies.
   - Download and extract the dataset.
   - Preprocess the data.
   - Train the models (Extract Features and Fine-Tuned).
   - Evaluate the models on the test set.
   - Visualize results.

## Model Architecture

The project uses VGG16, a pre-trained convolutional neural network, for transfer learning. Two approaches are implemented:

1. **Extract Features Model**:
   - Base VGG16 layers are frozen (non-trainable).
   - Custom dense layers: 512 units (ReLU), Dropout (0.3), 512 units (ReLU), Dropout (0.3), and a sigmoid output layer.
   - Trained to classify images using features extracted by VGG16.

2. **Fine-Tuned Model**:
   - Similar architecture to the Extract Features model.
   - The last convolutional layer (`block5_conv3`) of VGG16 is made trainable for fine-tuning.
   - Improves performance by adapting VGG16 weights to the waste classification task.

## Training

- Batch Size: 32
- Epochs: 10
- Optimizer: RMSprop (learning rate = 1e-4)
- Loss Function: Binary Crossentropy
- Callbacks:
  - Early Stopping (monitor: validation loss, patience: 4)
  - Model Checkpoint (save best model based on validation loss)
  - Custom Learning Rate Scheduler (exponential decay)
- Data Augmentation:
  - Rescale: 1/255
  - Width/Height Shift: 0.1
  - Horizontal Flip: True

Training involves two phases:

1. Training the Extract Features model with frozen VGG16 layers.
2. Fine-tuning by unfreezing the last convolutional layer.

## Evaluation

Both models are evaluated on the test set using:

- Classification Report: Precision, recall, F1-score, and accuracy for each class (O, R).
- Loss and Accuracy Curves: Plotted for training and validation sets.
- Sample Predictions: Visualized with actual vs. predicted labels.

## Results

- Extract Features Model: Provides baseline performance with good accuracy but may struggle with complex patterns.
- Fine-Tuned Model: Typically achieves higher accuracy due to task-specific weight adjustments in the VGG16 layer.
- Detailed classification reports are printed during evaluation, showing performance metrics for both models.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

Ensure your code follows the project's coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
