# Enhanced Climate Chatbot
import streamlit as st
import nltk
import string
import numpy as np
import re
import time
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import pandas as pd

# Download required NLTK data with better error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling."""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                print(f"Downloaded NLTK package: {package}")
            except Exception as e:
                if package == 'punkt_tab':
                    # Fallback to regular punkt
                    try:
                        nltk.download('punkt', quiet=True)
                        print("Using punkt instead of punkt_tab")
                    except:
                        pass
                else:
                    print(f"Could not download {package}: {e}")

# Call the download function
download_nltk_data()

# --- Load text file ---
def load_text_file(file_path: str) -> list:
    """Load and tokenize text file into sentences."""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return sent_tokenize(raw_text)

# --- Enhanced Preprocess sentences ---
def preprocess(sentences: list) -> list:
    """Enhanced preprocessing with lemmatization and better cleaning."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = []
    
    for sent in sentences:
        # Clean and normalize text
        sent = re.sub(r'[^\w\s]', ' ', sent.lower())
        sent = re.sub(r'\s+', ' ', sent).strip()
        
        tokens = word_tokenize(sent)
        # Remove stop words and lemmatize
        tokens = [lemmatizer.lemmatize(t) for t in tokens 
                 if t not in stop_words and len(t) > 2]
        processed.append(" ".join(tokens))
    return processed

# --- Get multiple relevant sentences with confidence scores ---
def get_relevant_sentences_with_confidence(query: str, sentences: list, processed: list, top_k: int = 3):
    """Get top-k most relevant sentences with confidence scores."""
    if not query.strip():
        return [], []
    
    # Preprocess query
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    query_clean = re.sub(r'\s+', ' ', query_clean).strip()
    query_tokens = word_tokenize(query_clean)
    query_processed = " ".join([lemmatizer.lemmatize(t) for t in query_tokens 
                               if t not in stop_words and len(t) > 2])
    
    texts = processed + [query_processed]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    # Get top-k results
    scores = cosine_sim[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = [scores[i] for i in top_indices]
    top_sentences = [sentences[i] for i in top_indices]
    
    return top_sentences, top_scores

# --- Enhanced Chatbot function ---
def enhanced_chatbot(query: str, sentences: list, processed: list):
    """Enhanced chatbot with confidence scoring and multiple responses."""
    responses, scores = get_relevant_sentences_with_confidence(query, sentences, processed, top_k=3)
    
    if not responses:
        return {
            'primary_response': "I'm sorry, I don't understand your question. Could you please rephrase it?",
            'confidence': 0.0,
            'alternatives': [],
            'suggestions': ["What is climate change?", "How does climate change affect the environment?", "What can we do about climate change?"]
        }
    
    primary_confidence = scores[0]
    
    # Generate response based on confidence
    if primary_confidence > 0.3:
        primary_response = responses[0]
        alternatives = responses[1:] if len(responses) > 1 else []
    elif primary_confidence > 0.1:
        primary_response = f"I found some relevant information, though I'm not entirely certain: {responses[0]}"
        alternatives = responses[1:] if len(responses) > 1 else []
    else:
        primary_response = "I couldn't find a confident answer to your question. Here are some topics I can help with:"
        alternatives = [
            "Climate change causes and effects",
            "Environmental impacts of climate change", 
            "Solutions to address climate change",
            "Individual actions to combat climate change"
        ]
    
    return {
        'primary_response': primary_response,
        'confidence': primary_confidence,
        'alternatives': alternatives,
        'suggestions': ["What causes climate change?", "How can individuals help?", "What are the effects of climate change?"]
    }

# --- Enhanced chatbot with sentiment analysis and response ranking ---
def analyze_query_intent(query: str) -> dict:
    """Analyze the intent and sentiment of the user's query."""
    query_lower = query.lower()
    
    # Intent classification
    if any(word in query_lower for word in ['what', 'define', 'explain', 'mean']):
        intent = 'definition'
    elif any(word in query_lower for word in ['how', 'way', 'method', 'solution']):
        intent = 'how_to'
    elif any(word in query_lower for word in ['why', 'reason', 'cause']):
        intent = 'causation'
    elif any(word in query_lower for word in ['when', 'time', 'year', 'future']):
        intent = 'temporal'
    elif any(word in query_lower for word in ['where', 'location', 'place']):
        intent = 'location'
    else:
        intent = 'general'
    
    # Sentiment analysis (simple)
    positive_words = ['good', 'help', 'solution', 'positive', 'benefit']
    negative_words = ['bad', 'problem', 'issue', 'negative', 'damage', 'harm']
    
    pos_count = sum(1 for word in positive_words if word in query_lower)
    neg_count = sum(1 for word in negative_words if word in query_lower)
    
    if pos_count > neg_count:
        sentiment = 'positive'
    elif neg_count > pos_count:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {'intent': intent, 'sentiment': sentiment}

# --- Response quality scoring ---
def score_response_quality(query: str, response: str, confidence: float) -> dict:
    """Score the quality of a response based on various factors."""
    scores = {}
    
    # Length appropriateness (not too short, not too long)
    response_length = len(response.split())
    if 20 <= response_length <= 150:
        scores['length'] = 1.0
    elif 10 <= response_length < 20 or 150 < response_length <= 200:
        scores['length'] = 0.7
    else:
        scores['length'] = 0.3
    
    # Keyword overlap
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
    scores['relevance'] = min(overlap * 2, 1.0)
    
    # Confidence contribution
    scores['confidence'] = confidence
    
    # Overall quality score
    overall_quality = (scores['length'] * 0.3 + scores['relevance'] * 0.4 + scores['confidence'] * 0.3)
    
    return {
        'overall': overall_quality,
        'length_score': scores['length'],
        'relevance_score': scores['relevance'],
        'confidence_score': scores['confidence']
    }

# --- Export conversation history ---
def export_conversation_history():
    """Export conversation history as JSON."""
    if st.session_state.conversation_history:
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_queries': st.session_state.total_queries,
            'conversations': st.session_state.conversation_history
        }
        return json.dumps(export_data, indent=2)
    return None

# --- Conversation memory ---
def init_session_state():
    """Initialize session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0

def add_to_history(query: str, response: dict):
    """Add conversation to history."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.conversation_history.append({
        'timestamp': timestamp,
        'query': query,
        'response': response
    })
    st.session_state.total_queries += 1

# --- Enhanced Streamlit UI ---
def main():
    # MUST be the first Streamlit command
    st.set_page_config(
        page_title="Enhanced Climate Chatbot", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Display NLTK status
    def check_nltk_status():
        """Check and display NLTK package status."""
        required_packages = ['punkt', 'stopwords', 'wordnet']
        missing_packages = []
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
            except LookupError:
                missing_packages.append(package)
        
        if missing_packages:
            st.info(f"NLTK packages ready. Missing: {', '.join(missing_packages)} (will download automatically)")
        
    check_nltk_status()
    
    # Sidebar
    with st.sidebar:
        st.title("🌍 Chatbot Info")
        st.markdown("---")
        
        # Stats
        st.metric("Total Queries", st.session_state.total_queries)
        
        # Settings
        st.subheader("⚙️ Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_alternatives = st.checkbox("Show alternative responses", value=True)
        
        # Quick questions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What is climate change?",
            "What causes climate change?", 
            "How can we reduce climate change?",
            "What are the effects of climate change?",
            "How does climate change affect humans?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.quick_question = question
          # Export conversation history
        if st.session_state.conversation_history:
            st.subheader("💾 Export")
            if st.button("📥 Export Chat History"):
                export_data = export_conversation_history()
                if export_data:
                    st.download_button(
                        label="Download JSON",
                        data=export_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        # Conversation History
        if st.session_state.conversation_history:
            st.subheader("📝 Recent Conversations")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"{conv['timestamp']} - Query {len(st.session_state.conversation_history)-i}"):
                    st.text(f"Q: {conv['query']}")
                    st.text(f"A: {conv['response']['primary_response'][:100]}...")
                    if show_confidence:
                        st.text(f"Confidence: {conv['response']['confidence']:.2%}")
                        
        # Analytics
        if st.session_state.conversation_history:
            st.subheader("📊 Analytics")
            df = pd.DataFrame(st.session_state.conversation_history)
            if not df.empty:
                avg_confidence = df['response'].apply(lambda x: x['confidence']).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Confidence distribution chart
                confidences = df['response'].apply(lambda x: x['confidence']).tolist()
                fig = px.histogram(x=confidences, nbins=10, title="Confidence Distribution")
                st.plotly_chart(fig, use_container_width=True)

    # Main content
    st.title("🤖 Enhanced Climate Chatbot")
    st.markdown("Ask me anything about climate change! I'll provide detailed responses with confidence scores.")
    
    # Load knowledge base
    file_path = "climate_change.txt"
    try:
        with st.spinner("Loading knowledge base..."):
            sentences = load_text_file(file_path)
            processed = preprocess(sentences)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"📚 {len(sentences)} sentences loaded")
        with col2:
            st.info(f"🔍 {len(processed)} processed")
        with col3:
            st.info(f"💬 {st.session_state.total_queries} total queries")

        # Handle quick question
        query = ""
        if hasattr(st.session_state, 'quick_question'):
            query = st.session_state.quick_question
            del st.session_state.quick_question
          # Main input
        query = st.text_input(
            "Ask a question about climate change:",
            value=query,
            placeholder="e.g., What are the main causes of climate change?"
        )
        
        if query:
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # Analyze query intent
                query_analysis = analyze_query_intent(query)
                
                response = enhanced_chatbot(query, sentences, processed)
                response_time = time.time() - start_time
                
                # Score response quality
                quality_score = score_response_quality(query, response['primary_response'], response['confidence'])
                
                # Add to history
                add_to_history(query, response)
            
            # Display response
            st.markdown("---")
            
            # Query analysis
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Query Intent:** {query_analysis['intent'].title()}")
            with col2:
                st.info(f"**Query Sentiment:** {query_analysis['sentiment'].title()}")
            
            # Main response
            st.markdown("### 🤖 Response:")
            st.markdown(f"**{response['primary_response']}**")
            
            # Response metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                if show_confidence:
                    confidence_color = "green" if response['confidence'] > 0.3 else "orange" if response['confidence'] > 0.1 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{response['confidence']:.2%}]")
            with col2:
                st.markdown(f"**Response time:** {response_time:.2f}s")
            with col3:
                quality_color = "green" if quality_score['overall'] > 0.7 else "orange" if quality_score['overall'] > 0.4 else "red"
                st.markdown(f"**Quality Score:** :{quality_color}[{quality_score['overall']:.2%}]")
            
            # Alternative responses
            if show_alternatives and response['alternatives']:
                st.markdown("### 🔍 Alternative responses:")
                for i, alt in enumerate(response['alternatives'][:2], 1):
                    with st.expander(f"Alternative {i}"):
                        st.write(alt)
            
            # Suggestions
            if response['suggestions']:
                st.markdown("### 💡 Related questions you might ask:")
                cols = st.columns(len(response['suggestions']))
                for i, suggestion in enumerate(response['suggestions']):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.quick_question = suggestion
                            st.rerun()
            
            # Feedback
            st.markdown("---")
            st.markdown("### 📊 Was this response helpful?")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 Yes", key="helpful_yes"):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 No", key="helpful_no"):
                    st.error("Sorry! Try rephrasing your question.")
            with col3:
                if st.button("🔄 Clear History", key="clear_history"):
                    st.session_state.conversation_history = []
                    st.session_state.total_queries = 0
                    st.success("History cleared!")
                    st.rerun()

    except FileNotFoundError:
        st.error("❌ Could not find 'climate_change.txt'. Make sure it's in the project folder.")
        st.info("Please ensure the knowledge base file exists to use the chatbot.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your setup and try again.")

if __name__ == '__main__':
    main()
