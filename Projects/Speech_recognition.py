import streamlit as st
import speech_recognition as sr
import os
from datetime import datetime

def transcribe_speech(api_choice, language_code):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Please say something...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Processing your audio...")
            if api_choice == "Google":
                return recognizer.recognize_google(audio, language=language_code)
            elif api_choice == "Sphinx":
                return recognizer.recognize_sphinx(audio, language=language_code)
            else:
                st.error("Unsupported API choice.")
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from the service; {e}")
        except sr.WaitTimeoutError:
            st.error("Listening timed out while waiting for phrase to start.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
def save_to_file(text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.txt"
    with open(filename, "w") as file:
        file.write(text)
    return filename
def main():
    st.title("Speech Recognition App")
    api_choice = st.selectbox("Choose API", ["Google", "Sphinx"])
    st.markdown("### 2. Select the language")
    language_dict = {
        "English": "en-US",
        "French": "fr-FR",
        "Spanish": "es-ES",
        "German": "de-DE",
        "Italian": "it-IT",
        "Arabic": "ar-SA",
        "Chinese": "zh-CN",
    }
    language_name = st.selectbox("Language", list(language_dict.keys()))
    language_code = language_dict[language_name]

    if 'transcription' not in st.session_state:
        st.session_state['transcription'] = ''

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_speech(api_choice, language_code)
            if transcription:
                st.session_state['transcription'] = transcription
                st.success("Transcription successful!")
                st.write(transcription)

    if st.button("Save to File"):
        transcription = st.session_state.get('transcription', '')
        if transcription:
            filename = save_to_file(transcription)
            st.success(f"Transcription saved to {filename}")
        else:
            st.error("No transcription available to save.")
    
if __name__ == "__main__":
    main()
