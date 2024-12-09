import re
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
from langchain.memory import ConversationBufferMemory , ConversationBufferWindowMemory
from langchain_community.llms import Ollama


# Set up Streamlit app title
st.title("ChatGPT")

# Create a ConversationChain
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     prompt=prompt,
#     input_key="text",  # Explicitly set the key for user input
# )


# Initialize session state for messages and responses
if "messages" not in st.session_state:
    st.session_state.messages = []
if "responses" not in st.session_state:
    st.session_state.responses = []
# load the ML model
if "ML_Model" not in st.session_state:
    st.session_state.ML_Model = load_model('My_Best_Pipeline')

# load the LLm model
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    input_key="text",
    )

def encode_df(df):
    """
    This function encodes categorical variables in the input dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing customer data. The dataframe should have the following categorical columns:
        'Dependents', 'gender', 'Is_Married', 'Senior_Citizen'.

    Returns:
    pandas.DataFrame: The input dataframe with categorical columns encoded using binary mapping for 'Dependents', 'gender', 'Is_Married', 'Senior_Citizen'
    and label encoding for other categorical columns.
    """
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Is_Married'] = df['Is_Married'].map({'Yes': 1, 'No': 0})
    df['Senior_Citizen '] = df['Senior_Citizen '].map({'Yes': 1, 'No': 0})
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df


def scale_df(df):
    """
    This function scales the numerical columns in the input dataframe using the StandardScaler.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing customer data. The dataframe should have the following numerical columns:
        'tenure', 'Monthly_Charges', 'Total_Charges'.

    Returns:
    pandas.DataFrame: The input dataframe with the numerical columns scaled using StandardScaler.
    """
    scaler = StandardScaler()
    numerical_columns = ["tenure", 'Monthly_Charges', 'Total_Charges']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


def ml_model(df):
    """
    This function preprocesses the input dataframe, encodes categorical variables, scales numerical variables,
    and then makes a prediction using a pre-trained machine learning model.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing customer data. The dataframe should have the following columns:
        'gender', 'Senior_Citizen', 'Is_Married', 'Dependents', 'tenure', 'Phone_Service', 'Dual', 'Internet_Service',
        'Online_Security', 'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies',
        'Contract', 'Paperless_Billing', 'Payment_Method', 'Monthly_Charges', 'Total_Charges'.

    Returns:
    numpy.ndarray: A 1D array containing the predicted churn values for the input customers.
    """
    df = encode_df(df)
    df = scale_df(df)
    return st.session_state.ML_Model.predict(df)



# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # st.write("Hi there can you provide my some data like gender Senior_Citizen Is_Married ..")
        st.markdown(message["content"])

    # st.session_state.messages.append({
    #     "role": "assistant",
    #     "content": (
    #         "Can you provide me with some information such as: \n"
    #         "- gender \n"
    #         "- Senior_Citizen \n"
    #         "- Is_Married \n"
    #         "- Dependents \n"
    #         "- tenure \n"
    #     )
    # })
# Accept user input
if user_input := st.chat_input("Enter your query here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)



    with st.chat_message("assistant"):
        st.write("now i am processing your data...")

        response =  st.session_state.conversation.run({'text' : user_input})

        try:

            output = parser.invoke([response])

            df  = pd.DataFrame([output])

            df.rename(columns={"Senior_Citizen": "Senior_Citizen "}, inplace=True)

            churn_prediction = ml_model(df.iloc[[-1]])
            
            df.loc[len(st.session_state.df) - 1, "Churn"] = churn_prediction[0]
            

            st.dataframe(df)
            # st.session_state.responses.append(response)
            # st.session_state.messages.append({"role": "assistant", "content": response})
        # Display the response
        except Exception:
            pass
            # Handle any errors in the ML model encoding
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Handle any errors in the ML model prediction
            # error_message = f"There are missing value : {Exception} please add this information"
            # st.markdown(error_message)
            # st.session_state.messages.append({"role": "assistant", "content": error_message})