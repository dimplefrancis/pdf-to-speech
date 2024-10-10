#import setup_env
#setup_env.ensure_environment()

import boto3
from botocore.exceptions import ClientError
import json
import os
import anthropic
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TCON, TDRC
import PyPDF2
import io
import streamlit as st
import ipywidgets as widgets
from IPython.display import display
import subprocess

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def get_secret():
    secret_name = "anthropic_api"
    region_name = "eu-west-2"

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print(f"Error retrieving secret: {e}")
        raise e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)['ANTHROPIC_API_KEY']

# Retrieve the API key from Secrets Manager and set it as an environment variable
try:
    os.environ['ANTHROPIC_API_KEY'] = get_secret()
    print("API key retrieved and set successfully.")
except ClientError as e:
    print(f"Failed to retrieve and set API key: {e}")

# Create the Anthropic client using the retrieved API key
try:
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    print("Anthropic client created successfully.")
except Exception as e:
    print(f"Error creating Anthropic client: {e}")

# Initialize Polly client
try:
    polly_client = boto3.client('polly', region_name='us-west-2')
    print("Amazon Polly client created successfully.")
except Exception as e:
    print(f"Error creating Amazon Polly client: {e}")

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def text_to_speech_polly(text, filename, voice_id='Emma', output_format='mp3'):
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat=output_format,
            VoiceId=voice_id,
            Engine='neural'
        )
        
        if "AudioStream" in response:
            with open(filename, 'wb') as file:
                file.write(response['AudioStream'].read())
            print(f"Audio saved successfully: {filename}")
        else:
            print(f"Could not find AudioStream in the response for file: {filename}")
            
    except Exception as error:
        print(f"Error processing text to speech for file {filename}: {error}")

def extract_topic(text, num_words=3):
    try:
        vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
        dtm = vectorizer.fit_transform([text])
        
        lda = LatentDirichletAllocation(n_components=1, random_state=42)
        lda.fit(dtm)
        
        words = vectorizer.get_feature_names_out()
        topic_words = [words[i] for i in lda.components_[0].argsort()[:-num_words - 1:-1]]
        return " ".join(topic_words)
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return "Latest topic"

def add_metadata(file_path, title, album, artist, genre, year):
    audio = MP3(file_path, ID3=ID3)
    try:
        audio.add_tags()
    except:
        pass
    audio.tags.add(TIT2(encoding=3, text=title))
    audio.tags.add(TALB(encoding=3, text=album))
    audio.tags.add(TPE1(encoding=3, text=artist))
    audio.tags.add(TCON(encoding=3, text=genre))
    audio.tags.add(TDRC(encoding=3, text=str(year)))
    audio.save()

def create_podcasts(summarized_data, output_dir='podcasts'):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    
    outro_text = "Thanks for listening to the Empowered Minds Podcast. Don't forget to subscribe for more insights."
    
    for i, text in enumerate(summarized_data):
        episode_number = i + 1
        filename = os.path.join(output_dir, f'podcast_episode_{episode_number}.mp3')
        
        try:
            topic = extract_topic(text)
        except Exception as e:
            print(f"Error extracting topic for episode {episode_number}: {e}")
            topic = "General topic"
        
        intro_text = f"Welcome to episode {episode_number} of the Empowered Minds Podcast. In this episode, we'll be discussing {topic}. Stay informed with the latest in technology."
        
        full_text = f"{intro_text} {text} {outro_text}"
        
        text_to_speech_polly(full_text, filename, voice_id='Emma')
        
        try:
            add_metadata(filename, 
                         title=f"Episode {episode_number}: {topic}",
                         album="Empowered Minds Podcast",
                         artist="Empowered Minds Team",
                         genre="Technology",
                         year=2024)
        except Exception as e:
            print(f"Error adding metadata to episode {episode_number}: {e}")
        
        audio_files.append(filename)
        print(f"Created episode {episode_number}: {topic}")
    
    return audio_files

def summarize_text(text):
    try:
        max_chunk_length = 4000
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {"role": "user", "content": f"Summarize the following text in about 200 words: {chunk}"}
                ]
            )
            summaries.append(response.content[0].text)
        
        return " ".join(summaries)
    except Exception as e:
        print(f"Error in summarization: {type(e).__name__} - {str(e)}")
        return ""

def process_pdfs(pdf_files):
    extracted_texts = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        extracted_texts.append(text)
    return extracted_texts

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def streamlit_app():
    try:
        st.title("PDF to Podcast Converter")

        num_pdfs = st.number_input("Enter the number of PDFs you want to convert", min_value=1, max_value=10, value=1)

        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) == num_pdfs:
            if st.button("Convert to Podcast"):
                extracted_texts = process_pdfs(uploaded_files)

                summarized_data = []
                for text in extracted_texts:
                    summary = summarize_text(text)
                    summarized_data.append(summary)

                audio_files = create_podcasts(summarized_data)

                for i, file in enumerate(audio_files):
                    st.audio(file, format='audio/mp3')
                    st.download_button(
                        label=f"Download Episode {i+1}",
                        data=open(file, 'rb'),
                        file_name=f"podcast_episode_{i+1}.mp3",
                        mime="audio/mp3"
                    )

        elif uploaded_files:
            st.warning(f"Please upload exactly {num_pdfs} PDF files.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error details: {e}")

def launch_streamlit():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streamlit_run.py')
    subprocess.Popen(['python', script_path])

def main():
    print("Launching PDF to Podcast Converter...")
    launch_streamlit()
    print("Streamlit app launched. Please check the SageMaker Studio interface for the app URL.")

if __name__ == "__main__":
    main()