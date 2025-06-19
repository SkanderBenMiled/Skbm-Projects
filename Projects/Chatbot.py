import nltk
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK resources if not already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Loading and preprocessing data
file_path = 'C:\\Users\\Skander BM\\Downloads\\Test_train_chatbot.txt'
if not os.path.exists(file_path):
    st.error(f"Data file not found: {file_path}")
    st.stop()
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()
# Tokenize the text into sentences
sentences = sent_tokenize(data)

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

def get_most_relevant_sentence(query):
    query = preprocess_text(query)
    min_distance = float('inf')
    most_relevant_sentence = ""
    for i, sentence in enumerate(preprocessed_sentences):
        distance = nltk.edit_distance(query, sentence)
        if distance < min_distance:
            min_distance = distance
            most_relevant_sentence = sentences[i]  # Return the original sentence
    return most_relevant_sentence

def chatbot_response(user_input):
    if user_input.lower() in ['exit', 'quit', 'bye']:
        return "Goodbye! Have a great day!"
    else:
        response = get_most_relevant_sentence(user_input)
        return response if response else "I'm sorry, I don't have an answer for that."

st.title("Chatbot")
st.write("Ask me anything!")
user_input = st.text_input("You: ")
if user_input:
    response = chatbot_response(user_input)
    st.write(f"Chatbot: {response}")