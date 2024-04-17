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

# Load models and utilities from local storage
@st.cache(allow_output_mutation=True)
def load_assets():
    model_path = "./model_files/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    image_processor = ViTImageProcessor.from_pretrained(model_path)
    translator = Translator()
    return model, tokenizer, image_processor, translator

model, tokenizer, image_processor, translator = load_assets()

def load_image(image_file):
    if isinstance(image_file, str):
        response = requests.get(image_file, stream=True)
        image = Image.open(response.raw)
    else:
        image = Image.open(image_file)
    return image

def generate_caption(image):
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

def caption_to_speech(caption, lang='en'):
    tts = gTTS(text=caption, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_audio_file_download_link(fp, filename):
    b64 = base64.b64encode(fp.getvalue()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download audio file</a>'
    return href

st.title("Image Captioning with Audio Output")
image_url = st.text_input("Enter Image URL", placeholder="Paste your image URL here...")
image_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])

if image_url or image_file:
    image = load_image(image_url if image_url else image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    caption = generate_caption(image)
    st.write("Generated Caption:", caption)

    lang = st.selectbox("Select Language for Translation", options=['en', 'hi', 'te', 'es', 'de', 'fr', 'it'])
    if st.button('Translate and Convert to Speech'):
        translated_caption = translate_text(caption, dest_language=lang)
        st.write("Translated Caption:", translated_caption)

        audio_fp = caption_to_speech(translated_caption, lang=lang)
        st.audio(audio_fp, format="audio/mp3")

        download_link = get_audio_file_download_link(audio_fp, "caption.mp3")
        st.markdown(download_link, unsafe_allow_html=True)
else:
    st.error("Please provide an image URL or upload an image.")


