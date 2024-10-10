import json
import streamlit as st
from sagemaker_inference import content_types, decoder
from latestpodcast import process_pdfs, summarize_text, create_podcasts
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    logger.info(f"Loading model from {model_dir}")
    # Load any necessary models or resources
    return "model"

def input_fn(input_data, content_type):
    logger.info(f"Received input with content type: {content_type}")
    if content_type == content_types.JSON:
        input_data = json.loads(input_data)
        logger.info(f"Parsed JSON input: {input_data}")
        return input_data
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    logger.info("Starting prediction")
    pdf_files = input_data['pdf_files']
    logger.info(f"Processing {len(pdf_files)} PDF files")
    extracted_texts = process_pdfs(pdf_files)
    logger.info("Texts extracted from PDFs")
    summarized_data = [summarize_text(text) for text in extracted_texts]
    logger.info("Texts summarized")
    audio_files = create_podcasts(summarized_data)
    logger.info(f"Created {len(audio_files)} audio files")
    return audio_files

def output_fn(prediction, accept):
    logger.info(f"Preparing output with accept type: {accept}")
    if accept == content_types.JSON:
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")

# Streamlit interface
def streamlit_interface():
    st.title("PDF to Podcast Converter")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Convert to Podcast"):
        audio_files = predict_fn({'pdf_files': uploaded_files}, None)
        for i, file in enumerate(audio_files):
            st.audio(file, format='audio/mp3')
            st.download_button(f"Download Episode {i+1}", file, f"podcast_episode_{i+1}.mp3", "audio/mp3")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8501))
    st.run(streamlit_interface, server_port=port, server_address='0.0.0.0')