## Handwritten Character Recognition

This repository contains a deep learning model for handwritten character recognition using the MNIST dataset. The model is implemented in TensorFlow and Keras, leveraging Convolutional Neural Networks (CNNs) for high-accuracy classification.

----

## Dataset

The project utilizes the MNIST dataset, stored in mnist-original.mat, which includes 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels in size. The dataset is preprocessed before training to enhance model performance.

----

## Dataset Preprocessing

-The pixel values are normalized to a range of 0 to 1.

-The dataset is split into training and testing sets.

-Labels are converted into categorical format for multi-class classification.

----

## Features

This repository includes:

-A TensorFlow and Keras-based deep learning model.

-Data preprocessing (normalization and reshaping) for efficient learning.

-A CNN architecture designed for digit recognition.

-Model training and validation with performance metrics (accuracy and loss).

-Visualizations for data samples, training progress, and predictions.

----

## Installation

To use this repository, follow these steps:

Clone the repository:

git clone https://github.com/yourusername/handwritten-character-recognition.git
cd handwritten-character-recognition

Install the required dependencies:

pip install -r requirements.txt

----

## Usage

**Running the Notebook**

To train and evaluate the model, open the Jupyter Notebook:

jupyter notebook "Handwritten Character Recognition.ipynb"

The notebook will:

Load the MNIST dataset.

Train a CNN model on the dataset.

Evaluate the model’s accuracy on test data.

Visualize sample predictions.

----

## Training the Model

-The CNN model consists of multiple convolutional layers followed by fully connected layers.

-The model is trained using the Adam optimizer with categorical cross-entropy loss.

-The training process includes real-time monitoring of accuracy and loss.

----

## Dependencies

Ensure that you have the following libraries installed:

TensorFlow: Deep learning framework for model training.

Keras: High-level neural network API.

NumPy: For numerical operations and data manipulation.

Matplotlib: For plotting graphs and visualizations.

scikit-learn: For dataset splitting and evaluation.

----

## Results

The trained model achieves high accuracy on the MNIST dataset. The notebook includes:

-Visualization of training and validation loss/accuracy.

-Sample predictions with actual vs. predicted labels.

-Performance metrics, including confusion matrices and classification reports.
