# 🎓 Admission Prediction Web App

This project is a Flask-based web application that predicts a student's chance of admission to a university based on input features such as GRE score, TOEFL score, CGPA, etc. The app uses a deep learning model and stores predictions in MongoDB for later retrieval and analysis.

---

## 🧠 Features

- Predicts probability of admission using a trained deep learning model.
- Scales features using a pre-trained `StandardScaler`.
- Accepts JSON input and returns a prediction.
- Stores each prediction in MongoDB.
- Includes a web-based form and an API endpoint for prediction.

---

## 📁 Project Structure

.
├── app.py # Main Flask app
├── model/
│ ├── admission_model.h5 # Trained Keras model
│ └── scaler.pkl # Scikit-learn StandardScaler
├── templates/
│ └── index.html # HTML form for web input
├── static/ # Optional: for CSS/JS if needed
├── requirements.txt # Required Python packages
└── README.md # This file


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/admission-predictor.git
cd admission-predictor
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Sample requirements.txt:
Flask
numpy
tensorflow
scikit-learn
joblib
pymongo

### MongoDB Setup
Make sure MongoDB is installed and running locally:
```bash
mongod --dbpath "C:/data/db"  # Adjust path as needed
```
Database: admissionDB
Collection: predictions

### Run the App
```bash
python app.py
```

Go to: http://127.0.0.1:5000