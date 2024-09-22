import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from transformers import BlipProcessor, BlipForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv
import torch
import io
import soundfile as sf
import torchaudio

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

urgency = ChatPromptTemplate.from_template(
"""
You are a complaint assistant. Your task is to analyse the complaint and determine whether it is emergency
or not. Always give answer in 'YES' and 'NO' only.
Complaint: {input}
"""
)
query = ChatPromptTemplate.from_template(
"""
You are a complaint assistant. Your task is to categorize user complaints into the following departments:
Railway Healthcare, Railway Police, Railway Engineer, Railway Food, Railway Staff, or Ticket collecting officer.
Also analyse the state of the complain need to be declare as emergency and if it need to be then tell the user it is an emergency complaint.
Keep it short about 50 words.
Based on the user's complaint, tell them which department it has been assigned to and respond with:
'Your complaint is registered with "Department name" and will be attended to shortly.'
Complaint: {input}
"""
)

def reply(complaint):
    main = query.invoke({'input':complaint})
    response = llm.invoke(main)
    department = response.content
    
    declaration = urgency.invoke({'input':complaint})
    urgent = llm.invoke(declaration)
    urgent_content = urgent.content
    
    return department, urgent_content

# BLIP for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Whisper for audio transcription using transformers
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

def transcribe_audio(audio_bytes):
    # Read the audio file using soundfile to get waveform and sample rate
    audio, original_samplerate = sf.read(io.BytesIO(audio_bytes))
    
    # If the original sample rate is not 16kHz, resample it
    if original_samplerate != 16000:
        audio = torch.tensor(audio).unsqueeze(0)  # Convert to a torch tensor with batch dimension
        audio = audio.double()  # Convert the tensor to float64 precision (double)
        resampler = torchaudio.transforms.Resample(orig_freq=original_samplerate, new_freq=16000)
        audio = resampler(audio).squeeze(0)  # Resample and remove the batch dimension
    
    # Preprocess the audio input for Whisper
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
    
    # Generate transcription
    predicted_ids = whisper_model.generate(inputs.input_features)
    transcription = whisper_processor.decode(predicted_ids[0], skip_special_tokens=True)
    
    return transcription



st.title("Railway Complaint Assistant Chatbot")

uploaded_image = st.file_uploader("Upload an image related to your complaint", type=["png", "jpg", "jpeg"])
caption = ""
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = generate_caption(image)
    st.write(f"Generated Caption: {caption}")

# Upload audio and get transcription
uploaded_audio = st.file_uploader("Upload an audio file for your complaint", type=["wav", "mp3", "m4a"])
transcription = ""
if uploaded_audio:
    st.audio(uploaded_audio)
    audio_bytes = uploaded_audio.read()
    
    # Transcribe audio using the Whisper model
    transcription = transcribe_audio(audio_bytes)
    st.write(f"Transcription: {transcription}")

# User input for the complaint
user_complaint = st.text_input("Enter or modify your complaint:", value=caption or transcription)

# Send button to process the complaint
send_button = st.button("Send")
if send_button:
    if user_complaint:
        department, urgent = reply(user_complaint)
        with st.container():
            st.write("**Complaint Response**")
            st.write(department)
            st.write("**Is this an emergency?**")
            st.write(urgent)
    else:
        st.warning("Please enter a complaint before submitting.")
