# Robo Advisor
- Encodes and preprocesses user data
- Trains a classification model on risk profiles
- Predicts a user’s **investment risk profile** (e.g., Conservative, Moderate, Aggressive)
- Suggests an **asset allocation strategy** based on the prediction

Built with **Python, scikit-learn, and Streamlit**.

# Features
- Data preprocessing & encoding (LabelEncoder / OrdinalEncoder)
- Model training and evaluation with stratified train-test split
- Risk profile prediction for new user inputs
- Human-readable decoding of predictions
- Interactive **Streamlit web app**


# How To Run
- pip install -r requirements.txt
- streamlit run app/app.py