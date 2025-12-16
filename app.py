import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- Load the saved model and vectorizer ---
# These files were created in Phase A.
# Ensure they are in the same folder as this script.
try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('sentiment_model.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run the 'train_and_save_model.py' script first.")
    st.stop()

# --- Re-create the preprocessing function ---
# This MUST be identical to the function used in the training script.
# Your training script used Lemmatization, so we use it here too.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# --- Set up the Streamlit page ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")
st.title("IMDB Movie Review Sentiment Analyzer")
st.write("This web app uses a pre-trained Machine Learning model to predict whether a movie review is positive or negative.")
st.write("Enter a review below and click 'Analyze Sentiment'.")

# --- Create the user interface elements ---
# Text area for user input
user_input = st.text_area(
    "Movie Review:", 
    "I saw this film last week and it was phenomenal! The acting was convincing and the plot was gripping from start to finish. I would highly recommend it to anyone.",
    height=150
)

# Button to trigger the prediction
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # 1. Preprocess the user's input text
        cleaned_input = preprocess_text(user_input)
        
        # 2. Transform the text using the loaded TF-IDF vectorizer
        input_vector = vectorizer.transform([cleaned_input])
        
        # 3. Predict the sentiment using the loaded model
        prediction = model.predict(input_vector)
        
        # 4. Display the result
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.success("This looks like a POSITIVE review! 😊")
        else:
            st.error("This looks like a NEGATIVE review. 😞")
    else:

        st.warning("Please enter a movie review to analyze.")
