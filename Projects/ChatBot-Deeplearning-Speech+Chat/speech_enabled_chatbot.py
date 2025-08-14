"""
Speech-Enabled Chatbot
This application combines speech recognition with a text-based chatbot to create
a conversational AI that can handle both text and voice input.
"""

import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpeechEnabledChatbot:
    def __init__(self, text_file_path):
        """Initialize the chatbot with text data and preprocessing components."""
        self.text_file_path = text_file_path
        self.sentences = []
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.microphone_available = False
        
        # Try to initialize microphone
        try:
            self.microphone = sr.Microphone()
            self.microphone_available = True
        except Exception as e:
            print(f"Warning: Microphone not available: {e}")
            self.microphone_available = False
        
        # Load and preprocess the text data
        self.load_and_preprocess_text()
        
    def load_and_preprocess_text(self):
        """Load text from file and preprocess it for the chatbot."""
        try:
            with open(self.text_file_path, 'r', encoding='utf-8') as file:
                text_data = file.read().lower()
            
            # Tokenize into sentences
            self.sentences = sent_tokenize(text_data)
            
            # Preprocess sentences
            processed_sentences = []
            for sentence in self.sentences:
                # Remove punctuation and numbers
                sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
                # Tokenize words
                words = word_tokenize(sentence)
                # Remove stopwords and stem
                stop_words = set(stopwords.words('english'))
                processed_words = [self.stemmer.stem(word) for word in words if word not in stop_words]
                processed_sentences.append(' '.join(processed_words))
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(processed_sentences)
            
            st.success("✅ Text data loaded and preprocessed successfully!")
            
        except FileNotFoundError:
            st.error(f"❌ Could not find the file: {self.text_file_path}")
        except Exception as e:
            st.error(f"❌ Error processing text data: {str(e)}")
    
    def preprocess_user_input(self, user_input):
        """Preprocess user input to match the training data format."""
        # Convert to lowercase and remove punctuation
        user_input = user_input.lower()
        user_input = re.sub(r'[^a-zA-Z\s]', '', user_input)
        
        # Tokenize and remove stopwords
        words = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        processed_words = [self.stemmer.stem(word) for word in words if word not in stop_words]
        
        return ' '.join(processed_words)
    
    def get_response(self, user_input):
        """Generate a response based on user input using cosine similarity."""
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # Preprocess user input
        processed_input = self.preprocess_user_input(user_input)
        
        if not processed_input.strip():
            return "I understand you're trying to communicate, but I need more specific words to help you. Could you rephrase your question?"
        
        # Transform user input using the same vectorizer
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity with all sentences
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        
        # Find the most similar sentence
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[0][most_similar_idx]
        
        # Set a threshold for similarity
        if similarity_score < 0.1:
            return "I'm sorry, I don't have enough information about that topic. Could you ask about something related to climate change?"
        
        # Return the most relevant sentence
        return self.sentences[most_similar_idx]
    
    def transcribe_speech(self):
        """Transcribe speech to text using speech recognition."""
        if not self.microphone_available:
            return "Error: No microphone available. Please connect a microphone and restart the application."
        
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                st.info("🎤 Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            st.info("🎤 Listening... Please speak now!")
            
            # Listen for speech
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            st.info("🔄 Processing speech...")
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.RequestError as e:
            return f"Could not request results from speech recognition service: {e}"
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio. Please try speaking more clearly."
        except sr.WaitTimeoutError:
            return "Listening timeout. Please try again."
        except Exception as e:
            return f"An error occurred during speech recognition: {e}"

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Speech-Enabled Chatbot",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Speech-Enabled Chatbot")
    st.markdown("---")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        ### How to use this chatbot:
        
        **Text Input:**
        - Type your question in the text box
        - Click 'Send Message' or press Enter
        
        **Voice Input:**
        - Make sure your microphone is connected
        - Click 'Start Voice Input'
        - Speak clearly when prompted
        - Wait for transcription
        
        **Tips:**
        - Speak clearly and avoid background noise
        - Ask questions related to climate change
        - Be patient during speech processing
        """)
        
        st.header("🔧 System Status")
        if 'chatbot' in st.session_state:
            st.success("✅ Chatbot initialized")
            if hasattr(st.session_state.chatbot, 'microphone_available'):
                if st.session_state.chatbot.microphone_available:
                    st.success("✅ Microphone available")
                else:
                    st.warning("⚠️ Microphone not available")
            else:
                st.warning("⚠️ Microphone status unknown")
        else:
            st.warning("⚠️ Chatbot not initialized")
    
    # Initialize the chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 Initializing chatbot..."):
            text_file_path = "Envir_Chatbot/climate_change.txt"
            st.session_state.chatbot = SpeechEnabledChatbot(text_file_path)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("💬 Text Input")
        user_text_input = st.text_area(
            "Type your message here:",
            height=100,
            placeholder="Ask me about climate change..."
        )
        
        text_button = st.button("📤 Send Message", key="text_button")
        
        if text_button and user_text_input:
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.get_response(user_text_input)
                st.session_state.chat_history.append({
                    'type': 'text',
                    'user': user_text_input,
                    'bot': response
                })
    
    with col2:
        st.header("🎤 Voice Input")
        
        # Check if microphone is available
        if 'chatbot' in st.session_state and hasattr(st.session_state.chatbot, 'microphone_available'):
            if not st.session_state.chatbot.microphone_available:
                st.warning("⚠️ Microphone not available. Please connect a microphone to use voice input.")
                st.button("🎙️ Start Voice Input", key="voice_button", disabled=True)
            else:
                st.write("Click the button below to start voice input:")
                voice_button = st.button("🎙️ Start Voice Input", key="voice_button")
                
                if voice_button:
                    with st.spinner("🎤 Processing voice input..."):
                        transcribed_text = st.session_state.chatbot.transcribe_speech()
                        
                        if not transcribed_text.startswith("Could not") and not transcribed_text.startswith("Sorry") and not transcribed_text.startswith("Listening") and not transcribed_text.startswith("An error") and not transcribed_text.startswith("Error:"):
                            # Successful transcription
                            st.success(f"🎯 You said: '{transcribed_text}'")
                            
                            # Get chatbot response
                            response = st.session_state.chatbot.get_response(transcribed_text)
                            st.session_state.chat_history.append({
                                'type': 'voice',
                                'user': transcribed_text,
                                'bot': response
                            })
                        else:
                            # Error in transcription
                            st.error(f"❌ {transcribed_text}")
        else:
            st.warning("⚠️ Chatbot not initialized yet.")
            st.button("🎙️ Start Voice Input", key="voice_button", disabled=True)
    
    # Display chat history
    st.markdown("---")
    st.header("💬 Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"{'🎤' if chat['type'] == 'voice' else '💬'} Conversation {len(st.session_state.chat_history) - i}", expanded=(i == 0)):
                st.markdown(f"**You ({chat['type']}):** {chat['user']}")
                st.markdown(f"**Bot:** {chat['bot']}")
    else:
        st.info("👋 No conversations yet. Start by typing a message or using voice input!")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
