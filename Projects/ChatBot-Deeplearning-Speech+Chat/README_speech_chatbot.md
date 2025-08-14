# Speech-Enabled Chatbot 🎤🤖

A sophisticated chatbot application that combines Natural Language Processing (NLP) with Speech Recognition to create an interactive conversational AI that can handle both text and voice input.

## 🌟 Features

- **Dual Input Methods**: Support for both text and voice input
- **Speech Recognition**: Real-time speech-to-text conversion using Google Speech Recognition
- **Intelligent Responses**: Uses TF-IDF vectorization and cosine similarity for context-aware responses
- **Interactive UI**: Beautiful Streamlit interface with real-time chat history
- **Climate Change Knowledge**: Pre-trained on climate change content for educational conversations
- **Error Handling**: Robust error handling for speech recognition and microphone issues

## 🛠️ Technologies Used

- **Python 3.11+**
- **Streamlit**: Web application framework
- **NLTK**: Natural Language Processing
- **SpeechRecognition**: Speech-to-text conversion
- **scikit-learn**: Machine learning for text similarity
- **PyAudio**: Audio I/O (for microphone input)

## 📋 Prerequisites

Before running the application, ensure you have:

1. **Python 3.11 or higher** installed
2. **A microphone** connected to your device (for voice input)
3. **Internet connection** (for Google Speech Recognition API)
4. **Virtual environment** activated (recommended)

## 🚀 Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd f:\SkanderUser\Documents\GitHub\Skbm-Projects
   ```

2. **Activate your virtual environment**:
   ```bash
   Skander.venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements_speech_chatbot.txt
   ```

## 🎯 Usage

### Running the Application

1. **Start the Streamlit application**:
   ```bash
   streamlit run speech_enabled_chatbot.py
   ```

2. **Open your web browser** and go to the URL displayed in the terminal (usually `http://localhost:8501`)

### Using the Chatbot

#### Text Input 💬
1. Type your question in the text area on the left
2. Click "Send Message" or press Enter
3. View the response in the chat history

#### Voice Input 🎤
1. Ensure your microphone is connected and working
2. Click "Start Voice Input" on the right side
3. Speak clearly when prompted
4. Wait for the transcription and response

### Example Interactions

**Text Examples:**
- "What is climate change?"
- "How can we reduce carbon emissions?"
- "Tell me about renewable energy"

**Voice Examples:**
- Speak naturally: "What causes global warming?"
- Ask questions: "How does deforestation affect climate?"

## 🧪 Testing

Run the test script to verify all components are working:

```bash
python test_chatbot.py
```

The test script will check:
- ✅ Package imports
- ✅ NLTK data availability
- ✅ Text file accessibility
- ✅ Speech recognition setup
- ✅ Basic chatbot functionality

## 📁 Project Structure

```
speech_enabled_chatbot.py      # Main application
test_chatbot.py               # Test script
requirements_speech_chatbot.txt # Package requirements
Envir_Chatbot/
  └── climate_change.txt      # Knowledge base text file
```

## 🔧 Configuration

### Microphone Setup
- **Windows**: Ensure microphone permissions are granted
- **Troubleshooting**: If microphone issues occur, check device settings

### Speech Recognition Settings
- **Timeout**: 10 seconds listening timeout
- **Phrase Limit**: 10 seconds maximum phrase length
- **Ambient Noise**: Automatic adjustment for background noise

## 🚨 Troubleshooting

### Common Issues

1. **"No microphone available"**
   - Check if microphone is connected
   - Verify microphone permissions
   - Restart the application

2. **"Could not understand audio"**
   - Speak more clearly
   - Reduce background noise
   - Move closer to microphone

3. **Import errors**
   - Ensure virtual environment is activated
   - Install missing packages: `pip install -r requirements_speech_chatbot.txt`

4. **NLTK data missing**
   - Run the test script to automatically download required data
   - Or manually: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

## 🎨 Features in Detail

### Natural Language Processing
- **Tokenization**: Breaks text into sentences and words
- **Preprocessing**: Removes stopwords, punctuation, and applies stemming
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **Cosine Similarity**: Finds most relevant responses

### Speech Recognition
- **Google Speech Recognition**: Cloud-based speech-to-text
- **Ambient Noise Adjustment**: Automatically adjusts for environment
- **Error Handling**: Graceful handling of recognition failures

### User Interface
- **Responsive Design**: Works on different screen sizes
- **Real-time Updates**: Live chat history and status updates
- **Visual Feedback**: Loading indicators and status messages

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- **NLTK**: Natural Language Toolkit
- **Streamlit**: Amazing web app framework
- **Google**: Speech Recognition API
- **scikit-learn**: Machine learning tools

---

**Happy Chatting! 🎉**

For questions or issues, please check the troubleshooting section or create an issue in the repository.
