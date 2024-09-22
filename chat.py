from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load environment variables
load_dotenv()

# Configure the Google Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

# Initialize BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to generate image caption using BLIP
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to get response from Gemini model
def get_gemini_response(question):
    system_message = """
    You are a complaint assistant. Your task is to categorize user complaints into the following departments:
    Railway Healthcare, Railway Police, Railway Engineer, Railway Food, Railway Staff, or Ticket collecting officer.
    Also analyse the state of the complain need to be declare as emergency and if it need to be then tell the user it is an emergency complain.
    Keep it short about 50 words.
    Based on the user's complaint, tell them which department it has been assigned to and respond with:
    'Your complaint is registered with {Department name} and will be attended to shortly.'
    """
    
    chat.send_message(system_message, stream=False)
    response = chat.send_message(question, stream=True)
    return response

# Streamlit app configuration
st.set_page_config(page_title="Complaint Assistant")

st.header("Railway Complaint Assistant")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File uploader for image
uploaded_image = st.file_uploader("Upload an image related to your complaint", type=["png", "jpg", "jpeg"])

# Process the image and display the generated caption
caption = ""
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = generate_caption(image)
    st.write(f"Generated Caption: {caption}")

# Provide a text input for the user to refine the query, pre-filling it with the caption if available
query = st.text_input("Modify your complaint or add more details:", value=caption)

# Submit button
submit = st.button("Submit Complaint")

# Handle the complaint submission
if submit and query:
    response = get_gemini_response(query)
    
    st.session_state['chat_history'].append(("You", query))
    
    st.subheader("The Response is:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display the chat history
st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
