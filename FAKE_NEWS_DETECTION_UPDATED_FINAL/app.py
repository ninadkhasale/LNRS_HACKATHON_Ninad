import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Function for stemming and preprocessing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Load models and vectorizers
fake_news_model = joblib.load('fake_news_model.pkl')
fake_news_vectorizer = joblib.load('vectorizer.pkl')
isot_model = joblib.load('ISOT_logistic_regression_model.pkl')
isot_vectorizer = joblib.load('ISOT_tfidf_vectorizer.pkl')

# Streamlit app title
st.title("📰 Fake News Detection App")

# Introduction section with description about fake news as a problem
st.markdown("""
## Why Fake News is a Growing Problem
Fake news has become a global issue, spreading misinformation quickly and influencing public opinion, politics, and even global events. The rise of social media platforms has made it easier for fake news to go viral, causing confusion, panic, or sometimes even violence. Detecting and combating fake news is crucial to maintaining the integrity of information.

This app helps identify whether a news snippet is **real** or **fake**, using machine learning models trained on two popular datasets.
""")

# Dataset descriptions
st.markdown("""
## Available Datasets for Fake News Detection

### 1. **Fake News Dataset**
- This dataset consists of articles labeled as **real** or **fake**, based on the content of the news articles. It has been widely used for training models to detect fake news.
  
### 2. **ISOT Fake News Dataset**
- This dataset, collected by the Information Security and Object Technology (ISOT) Research Lab, contains separate collections of real and fake news articles from various sources. It includes **separate categories** for real and fake news, providing diverse insights into different types of misinformation.

You can select one of these datasets for fake news detection below.
""")

# Add a sidebar for the dataset selection
st.sidebar.header("Select Dataset for Prediction")
dataset_option = st.sidebar.selectbox(
    'Choose which dataset you would like to use:',
    ('Fake News Dataset', 'ISOT Fake News Dataset')
)

# Add a brief description on the sidebar based on the dataset selected
if dataset_option == 'Fake News Dataset':
    st.sidebar.markdown("""
    **Fake News Dataset** contains labeled news articles that are either **real** or **fake**, allowing the model to identify and classify based on textual features.
    """)
else:
    st.sidebar.markdown("""
    **ISOT Fake News Dataset** is a collection of news articles from various sources, divided into real and fake categories, providing a broader context for detecting misinformation.
    """)

# Input for news snippet
st.markdown("## Enter a News Snippet for Prediction")
news_snippet = st.text_area("Paste the news snippet you want to check:", height=150)

# Function to predict using the selected model and vectorizer
def predict_news(news_snippet, model, vectorizer):
    # Preprocess the input news snippet
    processed_snippet = stemming(news_snippet)
    
    # Convert text to numerical data using the vectorizer
    vectorized_snippet = vectorizer.transform([processed_snippet])
    
    # Predict using the selected model
    prediction = model.predict(vectorized_snippet)
    
    # Return prediction result
    return 'Real' if prediction[0] == 0 else 'Fake'

# Predict button
if st.button("Check News"):
    if news_snippet.strip() != "":
        if dataset_option == 'Fake News Dataset':
            prediction = predict_news(news_snippet, fake_news_model, fake_news_vectorizer)
        else:
            prediction = predict_news(news_snippet, isot_model, isot_vectorizer)
        
        st.success(f"The news is **{prediction}**.")
    else:
        st.error("Please enter a valid news snippet.")

# Footer
st.markdown("---")
st.markdown("### Created by Ninad Khasale - AI & ML Project")
st.markdown("Learn more about fake news detection and AI models at [GitHub](https://github.com/your-profile-link).")
