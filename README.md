# LNRS_HACKATHON_NINAD_KHASALE
Fake News Detection System
Project Overview
This project aims to build a robust Fake News Detection System that uses both classical machine learning models and modern transformer-based techniques to classify news articles as either real or fake. Given the rise of misinformation, our system provides reliable predictions by leveraging both traditional approaches and advanced deep learning methods.

Key Features
Dual Approach:

Classical Method: Uses traditional machine learning models such as Logistic Regression, Support Vector Machines (SVM), Naive Bayes, and Random Forest. These models rely on feature extraction techniques like TF-IDF and Bag of Words (BoW) to detect fake news.
Modern Method: Utilizes a transformer-based model, RoBERTa (a variant of BERT), fine-tuned for fake news classification. This method integrates with Google Search API and PolitiFact fact-checking API for real-time validation of news articles.
NLP Preprocessing:

The system processes raw text using a pipeline of text cleaning steps, including lowercasing, removing special characters, URLs, punctuation, and numbers. It also performs tokenization and feature extraction using TF-IDF for classical models.
The modern method uses token embeddings with RoBERTa for deeper semantic understanding.
Real-Time Verification:

The modern approach adds a layer of reliability by querying Google Search for similar news articles from trusted sources like BBC.
It also scrapes recent fact-check articles from PolitiFact to verify the authenticity of the claims.
Technologies Used
Streamlit: For building an interactive and user-friendly multi-page web app.
Python Libraries:
Scikit-learn: Used for training classical models.
Hugging Face Transformers: For utilizing the RoBERTa model for text classification.
Joblib: For saving and loading machine learning models.
BeautifulSoup: For web scraping PolitiFact fact-checks.
Requests: For integrating Google Search API.
How It Works
Classical Method:

Users input a news headline or content.
The text is preprocessed (lowercased, cleaned) and transformed into a TF-IDF vector.
The classical machine learning model predicts whether the news is fake or real based on these features.
Modern Method:

Users provide both news headline and content.
The RoBERTa transformer model generates a prediction by analyzing the context of the news.
If the news is classified as real, the system fetches related articles from Google Search and looks for fact-checks on PolitiFact.

Installation and Setup

Clone the Repository:
https://github.com/ninadkhasale/LNRS_HACKATHON_Ninad.git
cd fake-news-detection

Install Dependencies: Install the required Python libraries by running:
pip install -r requirements.txt

Run the Streamlit App: Launch the app using:
streamlit run app.py

Future Improvements
Multilingual Support: Add support for detecting fake news across various languages.
Expanded Dataset: Integrate more datasets to improve model accuracy across different domains.
Real-Time Monitoring: Implement real-time monitoring of social media platforms to flag trending fake news.

Conclusion
This project is designed to be an end-to-end solution for fake news detection, combining the best of both classical machine learning and modern deep learning approaches. With real-time article matching and fact-checking, it aims to provide reliable and scalable results for various use cases, from media companies to individual users.
