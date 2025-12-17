# Handwritten-Digit-Classification-using-Deep-Learning-MNIST-Dataset-

📌 Project Overview
This project demonstrates the implementation of a Deep Learning model to classify handwritten digits (0–9) using the MNIST dataset.
The model is built using TensorFlow and Keras and follows a complete Deep Learning pipeline including data preprocessing, model training, evaluation, and prediction.

🎯 Objective

To design and train a neural network capable of recognizing handwritten digits.

To understand the fundamentals of Deep Learning model development using TensorFlow.

To achieve high accuracy on unseen handwritten digit images.

🧠 Algorithm & Techniques Used

Multi-Layer Perceptron (MLP)

Feedforward Neural Network

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Activation Functions:

ReLU (Hidden Layers)

Softmax (Output Layer)

📂 Dataset

MNIST Dataset

Total Images: 70,000

Training Samples: 60,000

Test Samples: 10,000

Image Size: 28 × 28 pixels (grayscale)

⚙️ Project Workflow

Import required libraries

Load MNIST dataset

Data preprocessing:

Normalization of pixel values (0–255 → 0–1)

One-hot encoding of labels

Model architecture design

Model compilation

Model training and validation

Model evaluation

Prediction on new samples

🏗️ Model Architecture

Input Layer: Flatten (28×28 → 784)

Hidden Layer 1: Dense (256 neurons, ReLU)

Hidden Layer 2: Dense (256 neurons, ReLU)

Output Layer: Dense (10 neurons, Softmax)

📊 Results

Achieved high accuracy on the MNIST test dataset.

The model is able to correctly predict handwritten digits with strong performance.

Demonstrates effective learning of image patterns using Deep Learning.

🛠️ Technologies Used

Python

TensorFlow

Keras

NumPy

Matplotlib

▶️ How to Run the Project

Clone the repository:

git clone https://github.com/your-username/mnist-deep-learning.git

Install required dependencies:

pip install tensorflow numpy matplotlib

Run the script:

python MNIST_DL.py
📁 Project Structure
├── MNIST_DL.py
├── README.md
🚀 Future Improvements

Implement Convolutional Neural Networks (CNNs) for higher accuracy

Add confusion matrix and performance visualization

Deploy the model using a web interface (Flask / Streamlit)
