import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Lazy load and cache the LLM model
@st.cache_resource
def load_llm():
    groq_api_key = os.getenv('GROQ_API_KEY')
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

llm = load_llm()

urgency = ChatPromptTemplate.from_template(
"""
You are a complaint assistant. Your task is to analyze the complaint and determine whether it is an emergency
or not. Always give answer in 'YES' and 'NO' only.
Complaint: {input}
"""
)

query = ChatPromptTemplate.from_template(
"""
You are a complaint assistant. Your task is to categorize user complaints into the following departments:
Railway Healthcare, Railway Police, Railway Engineer, Railway Food, Railway Staff, or Ticket collecting officer.
Also analyze the state of the complaint to declare it as an emergency and if it needs to be, tell the user it is an emergency complaint.
Keep it short, about 50 words.
Based on the user's complaint, tell them which department it has been assigned to and respond with:
'Your complaint is registered with "Department name" and will be attended to shortly.'
Complaint: {input}
"""
)

# Cache the reply function to avoid recalculating responses
@st.cache
def reply(complaint):
    main = query.invoke({'input': complaint})
    response = llm.invoke(main)
    department = response.content
    
    declaration = urgency.invoke({'input': complaint})
    urgent = llm.invoke(declaration)
    urgent_content = urgent.content
    
    return department, urgent_content

# Lazy load and cache the BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, blip_model

processor, blip_model = load_blip_model()

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit interface
st.title("Railway Complaint Assistant Chatbot")

uploaded_image = st.file_uploader("Upload an image related to your complaint", type=["png", "jpg", "jpeg"])
caption = ""
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = generate_caption(image)
    st.write(f"Generated Caption: {caption}")

chat_container = st.container()
user_complaint = st.text_input("Enter or modify your complaint:", value=caption)
send_button = st.button("Send")

if send_button:
    if user_complaint:
        department, urgent = reply(user_complaint)
        with chat_container:
            st.chat_message("user").write(user_complaint)
            st.chat_message("ai").write(department)
