import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page title
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

st.title("😊 Sentiment Analysis using Machine Learning")
st.write("Enter a sentence below to predict whether its sentiment is Positive or Negative.")

# User input
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)

        if prediction[0] == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")