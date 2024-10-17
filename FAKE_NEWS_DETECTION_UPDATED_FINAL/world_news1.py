import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from web_scraper import search_bbc_via_google, scrape_fact_checks

# Load the classical model using joblib
try:
    classical_model = joblib.load("model1.pkl")
    print("Classical model loaded successfully!")
except Exception as e:
    st.error(f"Error loading classical model: {e}")

# Load Hugging Face tokenizer and model for modern detection
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

# Classical Method Prediction
def predict_classical(text):
    prediction = classical_model.predict([text])[0]
    return "Fake" if prediction == 1 else "Real"

# Modern Method Prediction
def predict_modern(title, content):
    input_str = f"<title>{title}<content>{content}<end>"
    inputs = tokenizer.encode_plus(
        input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return {"Fake": probabilities[0], "Real": probabilities[1]}

# Classical Method Page
def classical_method_page():
    st.title("Classical Fake News Detection")
    st.markdown(
        """
        ### Overview:
        This page uses a **classical machine learning model** trained on a labeled dataset to detect fake news. 
        The dataset contains articles labeled as 'Fake' or 'Real,' covering topics such as politics, technology, 
        and world news. The machine learning model uses features extracted from the article content, such as:
        
        - Word frequency
        - Term frequency-inverse document frequency (TF-IDF)
        - Length of the headline or article

        ### Instructions:
        1. Enter a **news headline or article content** in the text area below.
        2. Click on the **Predict** button to classify the news as **Real** or **Fake**.
        """
    )

    user_input = st.text_area("Enter News Headline or Content")
    if st.button("Predict (Classical Method)"):
        if user_input:
            result = predict_classical(user_input)
            st.write(f"Prediction: **{result}**")
        else:
            st.warning("Please enter some text.")

# Modern Method Page
def modern_method_page():
    st.title("Modern Fake News Detection with APIs")
    st.markdown(
        """
        ### Overview:
        This page uses a **state-of-the-art transformer model** (based on Hugging Face’s `roberta-fake-news-classification`) 
        to detect fake news. In addition to the model's prediction, it integrates with external APIs to provide:
        
        - **Google search results** for similar articles from trusted sources (like BBC)
        - **Fact-checks from PolitiFact** to validate the news content

        ### Instructions:
        1. Enter a **news headline** and **content** in the text fields below.
        2. Click on **Predict** to classify the news as **Real** or **Fake**.
        3. If classified as **Real**, the system will search for matching articles and fact-checks.
        """
    )

    news_headline = st.text_input("Enter News Headline")
    news_content = st.text_area("Enter News Content")

    if st.button("Predict (Modern Method)"):
        if news_headline and news_content:
            result = predict_modern(news_headline, news_content)
            st.write("Prediction:")
            st.write(f"Fake News Probability: {result['Fake']:.2f}")
            st.write(f"Real News Probability: {result['Real']:.2f}")

            if result["Real"] > result["Fake"]:
                st.success("The news is likely real.")
                articles = search_bbc_via_google(news_headline)
                if articles:
                    st.write("Similar Articles Found:")
                    for article in articles:
                        st.write(f"- [{article['title']}]({article['url']})")
                else:
                    st.write("No matching articles found.")

                fact_checks = scrape_fact_checks()
                if fact_checks:
                    st.write("Recent Fact-Checks:")
                    for check in fact_checks:
                        st.write(f"- [{check['title']}]({check['url']})")
                else:
                    st.write("No recent fact-checks available.")
            else:
                st.error("This news is classified as fake.")
        else:
            st.warning("Please enter both headline and content.")

# Streamlit Multi-Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Classical Method", "Modern Method"])

if page == "Classical Method":
    classical_method_page()
elif page == "Modern Method":
    modern_method_page()
