import streamlit as st
from PIL import Image
import requests
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
import torch
from gtts import gTTS
import base64
from io import BytesIO
from googletrans import Translator

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and processors only once using Streamlit's session state
if 'model' not in st.session_state:
    st.session_state.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    st.session_state.tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    st.session_state.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    st.session_state.translator = Translator()

def load_image(image_file):
    """Loads an image from a URL or a file uploader."""
    if isinstance(image_file, str):
        response = requests.get(image_file, stream=True)
        image = Image.open(response.raw)
    else:
        image = Image.open(image_file)
    return image

def generate_caption(image):
    """Generates a caption for the image using preloaded model and tokenizer."""
    inputs = st.session_state.image_processor(images=image, return_tensors="pt").to(device)
    outputs = st.session_state.model.generate(**inputs)
    caption = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

def translate_text(text, dest_language):
    """Translates text to the specified language."""
    try:
        translation = st.session_state.translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        st.error(f"Failed to translate text due to: {e}")
        return ""

def caption_to_speech(caption, lang='en'):
    """Converts caption to speech and returns the audio file as a byte stream."""
    tts = gTTS(text=caption, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_audio_file_download_link(fp, filename):
    """Creates a download link for the audio file."""
    b64 = base64.b64encode(fp.getvalue()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download audio file</a>'
    return href

# Streamlit UI setup
st.title("Image Captioning with Audio Output")
image_url = st.text_input("Enter Image URL", placeholder="Paste your image URL here...")
image_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'], help="Upload a JPG or PNG image.")

if image_url or image_file:
    # Load and display the image only once
    if 'image' not in st.session_state or st.session_state.last_image_url != image_url:
        image = load_image(image_url if image_url else image_file)
        st.session_state.image = image
        st.session_state.last_image_url = image_url
        st.session_state.caption = generate_caption(image)
        st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)
    else:
        st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)

    lang = st.selectbox("Select Language for Translation", options=['en', 'hi', 'te', 'es', 'de', 'fr', 'it'], help="Choose a language to translate the caption.")
    if st.button('Translate and Convert to Speech'):
        translated_caption = translate_text(st.session_state.caption, dest_language=lang)
        st.write("Translated Caption:", translated_caption)

        audio_fp = caption_to_speech(translated_caption, lang=lang)
        st.audio(audio_fp, format="audio/mp3")

        download_link = get_audio_file_download_link(audio_fp, "caption.mp3")
        st.markdown(download_link, unsafe_allow_html=True)
else:
    st.error("Please provide an image URL or upload an image.")
