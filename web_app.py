# Import libraries
import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import spacy
import string
from keras.models import load_model

# Preprocessing incoming text
nlp = spacy.load('en_core_web_sm')

def preprocessor(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = ' '.join(tokens)
    return processed_text.lower()

# Load the saved parts
nb_model = joblib.load('mnb.pkl')
tfidf = joblib.load('vectorizer_nb.pkl')
vectorizer_nn = joblib.load('vectorizer_nn.pkl')
nn_model = load_model('nn_classifier.h5')

# Function to classify NaiveBayes model
def classify_nb(text):
    # prerpocess text
    text = preprocessor(text)
    # transform with tfidf
    vectorized_text = tfidf.transform([text])
    # Make prediction with saved MultinomialNB model
    prediction = nb_model.predict(vectorized_text)
    # Return predictions (ham = 0, spam = 1)
    return prediction[0]

# Function to classify NN model
def classify_nn(text):
    # preprocess text
    text = preprocessor(text)
    # transform to vectorized 
    vectorized_text = vectorizer_nn.transform([text])
    # Make prediction with nn
    prediction = nn_model.predict(vectorized_text)
    # Return prediction (ham = 0, spam = 1)
    return np.round(prediction)[0][1]

# Create Streamlit web app
def main():
    
    # Set page title
    st.title("Spam Classifier")
    
    # Create drop down for selecting classifier
    classifier = st.selectbox("Select the classifier:", ("Naive Bayes",  "Neural Network"))
    
    # Create a input text box for entering text
    input_text = st.text_input("Enter the text you want to classify:")
    
    # Create a result button:
    if st.button("Result"):
        if classifier == "Naive Bayes":
            prediction = classify_nb(input_text)
        else:
            prediction = classify_nn(input_text)
            
        if prediction == 0:
            st.write("This is NOT spam.")
        else:
            st.write("This is SPAM.")
            

if __name__ == '__main__':
    main()