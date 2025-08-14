"""
Demo script for the Speech-Enabled Chatbot
This script demonstrates how to use the chatbot components programmatically.
"""

from speech_enabled_chatbot import SpeechEnabledChatbot
import time

def demo_text_chatbot():
    """Demonstrate text-based chatbot functionality."""
    print("🤖 Text Chatbot Demo")
    print("=" * 40)
    
    # Initialize the chatbot
    print("Initializing chatbot...")
    chatbot = SpeechEnabledChatbot("Envir_Chatbot/climate_change.txt")
    
    # Demo questions
    questions = [
        "What is climate change?",
        "How can we reduce emissions?",
        "Tell me about renewable energy",
        "What are the effects of global warming?",
        "How does deforestation impact the environment?"
    ]
    
    for question in questions:
        print(f"\n🔍 Question: {question}")
        response = chatbot.get_response(question)
        print(f"🤖 Response: {response}")
        time.sleep(1)  # Small delay for readability

def demo_speech_recognition():
    """Demonstrate speech recognition functionality (without actual microphone)."""
    print("\n🎤 Speech Recognition Demo")
    print("=" * 40)
    
    # Initialize the chatbot
    chatbot = SpeechEnabledChatbot("Envir_Chatbot/climate_change.txt")
    
    # Check microphone availability
    if chatbot.microphone_available:
        print("✅ Microphone is available for speech input")
        print("Note: In a real scenario, you would speak into the microphone")
    else:
        print("⚠️ Microphone not available - this is normal in some environments")
        print("Speech recognition would work with a connected microphone")

def demo_preprocessing():
    """Demonstrate text preprocessing functionality."""
    print("\n🔧 Text Preprocessing Demo")
    print("=" * 40)
    
    chatbot = SpeechEnabledChatbot("Envir_Chatbot/climate_change.txt")
    
    # Test text preprocessing
    test_inputs = [
        "What is CLIMATE CHANGE???",
        "How can we reduce carbon emissions?!",
        "Tell me about renewable energy sources..."
    ]
    
    for text in test_inputs:
        processed = chatbot.preprocess_user_input(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()

def main():
    """Run all demos."""
    print("🎉 Speech-Enabled Chatbot Demo")
    print("=" * 50)
    
    try:
        demo_text_chatbot()
        demo_preprocessing()
        demo_speech_recognition()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\nTo run the full interactive application:")
        print("streamlit run speech_enabled_chatbot.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check if all dependencies are installed correctly.")

if __name__ == "__main__":
    main()
