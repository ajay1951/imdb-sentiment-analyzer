import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- Download NLTK Data Packages ---
# This is the simplest and most effective method for Streamlit Cloud.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Load the saved model and vectorizer ---
# These files were created in Phase A.
try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = job.load('sentiment_model.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please ensure they are in your GitHub repository.")
    st.stop()

# --- Re-create the preprocessing function ---
# This MUST be identical to the function used in the training script.
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
user_input = st.text_area(
    "Movie Review:",
    "I saw this film last week and it was phenomenal! The acting was convincing and the plot was gripping from start to finish. I would highly recommend it to anyone.",
    height=150
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)
        
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.success("This looks like a POSITIVE review! 😊")
        else:
            st.error("This looks like a NEGATIVE review. 😞")
    else:
        st.warning("Please enter a movie review to analyze.")
