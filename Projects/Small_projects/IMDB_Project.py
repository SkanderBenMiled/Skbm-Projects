import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import streamlit as st

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

nltk.download('stopwords')
nltk.download('punkt')
STOP = set(stopwords.words('english'))

RE_HTML = re.compile(r'<.*?>')
RE_URL = re.compile(r'https?://\S+|www\.\S+')

@st.cache_data
def load_data(path='IMDB Dataset.csv'):
    df = pd.read_csv(path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['review_len'] = df['review'].str.split().apply(len)
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = RE_HTML.sub(' ', text)
    text = RE_URL.sub(' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP]
    return ' '.join(tokens)

@st.cache_resource
def create_vectorizer(max_features=20000, ngram_range=(1,2)):
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=128):
    es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return history

# Streamlit app
st.title('IMDB Movie Review Sentiment - Streamlit')

st.markdown('A simple pipeline: EDA → Preprocessing → TF-IDF → Dense NN training and prediction')

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Data loaded.')

if st.checkbox('Show raw data (first 5 rows)'):
    st.write(df.head())

st.subheader('Dataset distribution')
fig, ax = plt.subplots()
sns.countplot(x='sentiment', data=df, ax=ax)
st.pyplot(fig)

st.subheader('Review length distribution')
fig2, ax2 = plt.subplots()
sns.histplot(df, x='review_len', hue='sentiment', bins=50, element='step', ax=ax2)
st.pyplot(fig2)

# Preprocessing options
st.sidebar.header('Preprocessing & Model')
max_features = st.sidebar.number_input('TF-IDF max_features', min_value=1000, max_value=50000, value=20000, step=1000)
ngram_min = st.sidebar.number_input('ngram min', min_value=1, max_value=2, value=1)
ngram_max = st.sidebar.number_input('ngram max', min_value=1, max_value=2, value=2)
ngram_range = (ngram_min, ngram_max)

test_size = st.sidebar.slider('Test set fraction', 0.05, 0.4, 0.2)
epochs = st.sidebar.slider('Epochs', 1, 50, 10)
batch_size = st.sidebar.selectbox('Batch size', [32, 64, 128, 256], index=2)

# Preprocess text
if st.button('Run preprocessing'):
    with st.spinner('Cleaning text...'):
        df['clean'] = df['review'].apply(clean_text)
    st.success('Preprocessing done')
    st.write('Sample cleaned reviews:')
    st.write(df[['review', 'clean']].head())

# Prepare data and vectorizer
if st.button('Prepare TF-IDF and split data'):
    if 'clean' not in df.columns:
        df['clean'] = df['review'].apply(clean_text)

    X = df['clean'].values
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    tfidf = create_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    st.session_state['X_train_tfidf'] = X_train_tfidf
    st.session_state['X_test_tfidf'] = X_test_tfidf
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['tfidf'] = tfidf

    st.success('TF-IDF prepared and data split')
    st.write('Input dim:', X_train_tfidf.shape[1])

# Train model
if st.button('Train model'):
    if 'X_train_tfidf' not in st.session_state:
        st.error('Prepare TF-IDF and split data first')
    else:
        input_dim = st.session_state['X_train_tfidf'].shape[1]
        model = build_model(input_dim)
        with st.spinner('Training model...'):
            history = train_model(model, st.session_state['X_train_tfidf'].toarray(), st.session_state['y_train'], epochs=epochs, batch_size=batch_size)
        st.success('Training finished')

        # Evaluate
        test_loss, test_acc = model.evaluate(st.session_state['X_test_tfidf'].toarray(), st.session_state['y_test'], verbose=0)
        y_pred_prob = model.predict(st.session_state['X_test_tfidf'].toarray()).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        st.write(f'Test Accuracy: {test_acc:.4f}')
        st.text('Classification report:')
        st.text(classification_report(st.session_state['y_test'], y_pred))
        st.write('ROC AUC:', roc_auc_score(st.session_state['y_test'], y_pred_prob))

        # Plot training curves
        fig3, ax3 = plt.subplots(1,2, figsize=(12,4))
        ax3[0].plot(history.history['loss'], label='Train Loss')
        ax3[0].plot(history.history['val_loss'], label='Val Loss')
        ax3[0].legend(); ax3[0].set_title('Loss')

        ax3[1].plot(history.history['accuracy'], label='Train Acc')
        ax3[1].plot(history.history['val_accuracy'], label='Val Acc')
        ax3[1].legend(); ax3[1].set_title('Accuracy')
        st.pyplot(fig3)

        # Save model and vectorizer
        model.save('imdb_sentiment_model.h5')
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(st.session_state['tfidf'], f)
        st.success('Model and vectorizer saved')
        st.session_state['model'] = model

# Load existing model
if st.button('Load saved model'):
    if os.path.exists('imdb_sentiment_model.h5') and os.path.exists('tfidf_vectorizer.pkl'):
        model = tf.keras.models.load_model('imdb_sentiment_model.h5')
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        st.session_state['model'] = model
        st.session_state['tfidf'] = tfidf
        st.success('Model and vectorizer loaded')
    else:
        st.error('Saved model or vectorizer not found')

# Single review prediction
st.subheader('Predict sentiment for a single review')
user_review = st.text_area('Enter a movie review to predict')
if st.button('Predict'):
    if 'model' not in st.session_state or 'tfidf' not in st.session_state:
        st.error('Model not loaded or trained. Use Train model or Load saved model first.')
    else:
        clean = clean_text(user_review)
        vec = st.session_state['tfidf'].transform([clean])
        prob = st.session_state['model'].predict(vec.toarray()).ravel()[0]
        label = 'positive' if prob > 0.5 else 'negative'
        st.write(f'Predicted sentiment: {label} (probability {prob:.3f})')

# Show confusion matrix if available
if 'y_test' in st.session_state and 'model' in st.session_state:
    y_prob = st.session_state['model'].predict(st.session_state['X_test_tfidf'].toarray()).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(st.session_state['y_test'], y_pred)
    st.subheader('Confusion Matrix')
    st.write(cm)