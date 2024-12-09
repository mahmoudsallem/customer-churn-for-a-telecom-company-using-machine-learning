import re
import json
import pandas as pd
import streamlit as st
from langchain_ollama import OllamaLLM
from pycaret.classification import *
from langchain_core.prompts import ChatPromptTemplate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser 
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from utails import parse , LLM_model , ML_model

# Define the Pydantic model for the JSON data
parser = parse()
# Initialize the Large language Model
conversation = LLM_model()

def update_json(data, file_path="customer_data.json"):
    """
    This function saves the provided data to a JSON file.

    Parameters:
    data (dict): The data to be saved in JSON format.
    file_path (str, optional): The path of the JSON file. Defaults to "customer_data.json".

    Returns:
    None

    This function attempts to open the specified file in write mode and write the provided data to it.
    If the file cannot be opened or written to, an error message is displayed using Streamlit's error function.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        st.success(f"Data saved to {file_path}")
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Required fields for the JSON object
required_fields = [
    "gender", "Senior_Citizen", "Is_Married", "Dependents", "tenure", "Phone_Service", 
    "Dual", "Internet_Service", "Online_Security", "Online_Backup", "Device_Protection", 
    "Tech_Support", "Streaming_TV", "Streaming_Movies", "Contract", "Paperless_Billing", 
    "Payment_Method", "Monthly_Charges", "Total_Charges"
]

# Ensure session state variables are initialized
if "messages" not in st.session_state:
    st.session_state.messages = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "response_data" not in st.session_state:
    st.session_state.response_data = {}
if "missing_features" not in st.session_state:
    st.session_state.missing_features = []

# Streamlit App
st.title("ChatGPT-like Interface")
st.write("Interact with the chatbot to extract features and save them to a JSON file.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if there's a user input
if user_input := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant's response
    response = conversation.run({"text": user_input})
    st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    try:
        # Parse the response
        response_data = parser.invoke(response)
        st.session_state.response_data.update(response_data)

        # Identify missing fields
        st.session_state.missing_features = [field for field in response_data if response_data.get(field) in [None, "", "Unknown"]]

        if st.session_state.missing_features:
            st.info(f"Missing fields: {', '.join(st.session_state.missing_features)}")

    except json.JSONDecodeError as e:
        st.error(f"Could not extract information from the response. Please try again. Error: {e}")

# Handle missing data input
if st.session_state.missing_features:
    field = st.session_state.missing_features[0]
    user_input = st.text_input(f"Please provide a value for '{field}':", key=f"input_{field}")
    if user_input:
        st.session_state.response_data[field] = user_input
        st.session_state.missing_features.pop(0)
        # Refresh the app
        
if st.session_state.response_data:
    if st.button("Show ML Prediction"):
        st.write( ML_model(st.session_state.response_data))


# Display all collected data and provide options to save or exit
if st.session_state.response_data and not st.session_state.missing_features:
    st.success("All required data collected!")
    if st.button("Show JSON"):
        st.json(st.session_state.response_data)
    if st.button("Save Data and Exit"):
        update_json(st.session_state.response_data)
        # Save previous messages before resetting session state
        previous_messages = st.session_state.messages.copy()
        st.session_state.messages = previous_messages  # Preserve previous messages
        st.session_state.response_data = {}
        st.session_state.missing_features = []
        # Refresh the app
