import streamlit as st
import joblib
import re

# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('ISOT_logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('ISOT_tfidf_vectorizer.pkl')

# Function to clean the input text
def clean_input(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Function to predict if the input news is fake or real
def predict_news(input_text):
    # Clean the input text
    cleaned_input = clean_input(input_text)
    
    # Vectorize the cleaned input using the TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([cleaned_input])
    
    # Make the prediction (returns 0 for fake, 1 for real)
    prediction = model.predict(input_tfidf)
    
    # Return the result
    if prediction == 0:
        return "The news is Fake."
    else:
        return "The news is Real."

# Streamlit UI components
st.title("Fake News Detection")

# Input field for news text
user_input = st.text_area("Enter a news article to check if it's real or fake:")

# Button to trigger prediction
if st.button('Check'):
    if user_input:
        result = predict_news(user_input)
        st.write(result)
    else:
        st.write("Please enter some text to check.")
