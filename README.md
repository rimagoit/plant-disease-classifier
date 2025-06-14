# 🌿 Plant Leaf Disease Classifier and Treatment Advisory using Machine Learning Techniques

This project is a machine learning-based web application that classifies plant leaf diseases from uploaded images and provides treatment suggestions. It is designed to assist farmers, agronomists, and researchers in identifying plant health issues early and recommending effective treatment options.

---

## 📌 Key Features

- ✅ Upload a plant leaf image and get disease classification
- 🧠 Deep Learning (CNN) model trained on labeled plant disease images
- 💊 Provides treatment guidance based on the predicted disease
- 🌐 Simple and interactive UI built with Streamlit
- 🧪 Real-world application of AI in agriculture

---

## 🛠️ Tech Stack

| Component        | Technology       |
|------------------|------------------|
| Programming Lang | Python           |
| ML Framework     | TensorFlow (Keras) |
| Frontend         | Streamlit        |
| Data Handling    | NumPy, Pandas    |
| Visualization    | Matplotlib, Seaborn |

---

## 🚀 How to Run the App Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-leaf-disease-classifier.git

Install required packages
 pip install -r requirements.txt
Run the Streamlit app
 streamlit run main.py

🌱 Sample Usage
Upload an image of a diseased plant leaf.

The model will predict the disease (e.g., Apple Cedar Rust).

Based on the prediction, the app will suggest a treatment (e.g., Apply sulfur-based fungicide).

📷 Dataset Information
The model was trained on the PlantVillage dataset, which contains 50,000+ labeled leaf images across various plant species and diseases.

🌐 Live Demo
🚀 Launch on Streamlit Cloud
(Replace with your actual app URL once deployed)

🤖 Model Overview
Model Type: CNN (Convolutional Neural Network)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy Achieved: ~95% on validation data

💊 Treatment Logic
The app uses a dictionary-based lookup system to recommend treatments based on the classified disease. Each disease is mapped to:

A brief explanation

Suggested treatment steps
