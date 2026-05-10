Diabetes Prediction ML Project
A machine learning project that predicts whether a person is likely to have diabetes based on health parameters. It includes a trained model and a simple web app built with Streamlit to make predictions interactively.

Tech Stack

Language: Python
ML Library: scikit-learn
Web App: Streamlit
Data Handling: pandas, numpy
Model Saving: joblib


Project Structure
diabetes_ml_project/
├── app/          # Streamlit web app
├── data/         # Dataset files
├── models/       # Saved trained models
├── notebooks/    # Jupyter notebooks for EDA and training
├── src/          # Core scripts (preprocessing, training, etc.)
├── requirements.txt
└── README.md

Getting Started
1. Clone the repository
bashgit clone https://github.com/Demonking100/diabetes_ml_project.git
cd diabetes_ml_project
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app/app.py

How It Works
The model is trained on health-related features like glucose level, BMI, age, blood pressure, etc. Once trained, it gets saved using joblib and loaded by the Streamlit app, where users can input their values and get an instant prediction.
