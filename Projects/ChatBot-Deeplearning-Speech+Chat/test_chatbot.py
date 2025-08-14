"""
Test script for the Speech-Enabled Chatbot
This script tests the individual components before running the full Streamlit app.
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition imported successfully")
    except ImportError as e:
        print(f"❌ SpeechRecognition import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_nltk_data():
    """Test if required NLTK data is available."""
    print("\nTesting NLTK data...")
    
    try:
        import nltk
        
        # Test punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt tokenizer data found")
        except LookupError:
            print("⚠️ NLTK punkt tokenizer data not found, downloading...")
            nltk.download('punkt')
            print("✅ NLTK punkt tokenizer data downloaded")
        
        # Test stopwords
        try:
            nltk.data.find('corpora/stopwords')
            print("✅ NLTK stopwords data found")
        except LookupError:
            print("⚠️ NLTK stopwords data not found, downloading...")
            nltk.download('stopwords')
            print("✅ NLTK stopwords data downloaded")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data test failed: {e}")
        return False

def test_text_file():
    """Test if the climate change text file exists and is readable."""
    print("\nTesting text file...")
    
    text_file_path = "Envir_Chatbot/climate_change.txt"
    
    if os.path.exists(text_file_path):
        print(f"✅ Text file found: {text_file_path}")
        
        try:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"✅ Text file readable, content length: {len(content)} characters")
                return True
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return False
    else:
        print(f"❌ Text file not found: {text_file_path}")
        return False

def test_speech_recognition():
    """Test if speech recognition components are available."""
    print("\nTesting speech recognition...")
    
    try:
        import speech_recognition as sr
        
        # Test recognizer
        recognizer = sr.Recognizer()
        print("✅ Speech recognizer created successfully")
        
        # Test microphone (this won't actually use the microphone)
        try:
            microphone = sr.Microphone()
            print("✅ Microphone interface created successfully")
        except Exception as e:
            print(f"⚠️ Microphone test failed (this might be normal if no microphone is connected): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        return False

def test_chatbot_basic_functionality():
    """Test basic chatbot functionality without Streamlit."""
    print("\nTesting basic chatbot functionality...")
    
    try:
        # Import the chatbot class
        from speech_enabled_chatbot import SpeechEnabledChatbot
        
        # Initialize chatbot
        text_file_path = "Envir_Chatbot/climate_change.txt"
        chatbot = SpeechEnabledChatbot(text_file_path)
        print("✅ Chatbot initialized successfully")
        
        # Test text response
        test_input = "What is climate change?"
        response = chatbot.get_response(test_input)
        print(f"✅ Text response generated: '{response[:100]}...'")
        
        return True
    except Exception as e:
        print(f"❌ Basic chatbot functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Speech-Enabled Chatbot Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_nltk_data,
        test_text_file,
        test_speech_recognition,
        test_chatbot_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your chatbot is ready to run.")
        print("\nTo start the chatbot, run:")
        print("streamlit run speech_enabled_chatbot.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
