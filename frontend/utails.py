import pandas as pd
import re
import streamlit as st
from langchain_ollama import OllamaLLM
from pycaret.classification import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.llms import Ollama


def parse():
    """
    This function parses customer information provided as a JSON string and validates it against a predefined schema.
    It uses the Pydantic library to define a data model (Info) and validate the input data against this model.

    Parameters:
    None

    Returns:
    parser (JsonOutputParser): An instance of JsonOutputParser initialized with the Info model.
    The parser can be used to parse and validate customer information in JSON format.
    """
    class Info(BaseModel):
        gender: str = Field(description="Valid answers are 'Male', 'male', 'MALE', 'Female', 'female', or 'FEMALE'", required=True)
        Senior_Citizen: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Is_Married: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Dependents: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        tenure: float = Field(description="Provide the tenure in months as a numeric value.", required=True)
        Phone_Service: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Dual: str = Field(description="Valid answers are 'Yes', 'yes', 'YES', 'No', 'no', 'NO', or 'No phone service'.", required=True)
        Internet_Service: str = Field(description="Valid answers are 'DSL', 'dsl', 'DSL', 'Fiber optic', 'fiber optic', or 'No', 'no', 'NO'.", required=True)
        Online_Security: str = Field(description="Valid answers are 'Yes', 'yes', 'YES', 'No', 'no', 'NO', or 'No internet service'.", required=True)
        Online_Backup: str = Field(description="Valid answers are 'Yes', 'yes', 'YES', 'No', 'no', 'NO', or 'No internet service'.", required=True)
        Device_Protection: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Tech_Support: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Streaming_TV: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Streaming_Movies: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Contract: str = Field(description="Valid answers are 'Month-to-Month', 'One Year', 'Two Year', or their case variants.", required=True)
        Paperless_Billing: str = Field(description="Valid answers are 'Yes', 'yes', 'YES' or 'No', 'no', 'NO'.", required=True)
        Payment_Method: str = Field(description="Valid answers include 'Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check', or their case variants.", required=True)
        Monthly_Charges: float = Field( description="Provide a numeric value (e.g., '59.99').", required=True)
        Total_Charges: float = Field(description="Provide a numeric value (e.g., '500.75').", required=True)

    parser = JsonOutputParser(pydantic_object=Info)

    return parser


def LLM_model():
    """
    This function initializes a language model-based chatbot to extract specific customer features from user-provided text.
    The chatbot uses a conversation chain with a memory mechanism to retain previously collected data.
    It follows a step-by-step approach to ensure all required features are extracted accurately.
    The chatbot uses a PromptTemplate with a custom template to guide the conversation and extract the required features.
    The function returns a ConversationChain object that can be used to interact with the chatbot.

    Parameters:
    None

    Returns:
    ConversationChain: A ConversationChain object initialized with the specified language model, memory, prompt, and input key.
    """

    # Define the template
    template = """
    You are a chatbot designed to extract specific features from user-provided text, validate the data, and save it in a JSON file. 
    Follow a step-by-step approach to ensure all required features are extracted accurately, leveraging memory to retain previously collected data.

    ### Context:
    - Conversation so far:
    {history}

    - User input:
    {text}

    ### Tasks:
    1. **Extract Data**:
    - Analyze the user's input to identify any information provided that matches the required features.
    - Save the extracted data in the correct JSON structure.

    2. **Check for Missing Features**:
    - Compare the extracted data with the full list of required features.
    - Identify any missing features and ask the user to provide the missing data.
    - Example: "I noticed you haven't mentioned your `Internet_Service`. Could you specify it?"

    3. **Save or Update JSON File**:
    - Save the data to a JSON file named `customer_data.json`.
    - Append newly provided data to the appropriate fields.
    - Overwrite any previously incorrect or incomplete values with the updated data.

    4. **Verify Completeness**:
    - Ensure the JSON object is valid and complete.
    - Respond to the user with the updated JSON object once all required fields are filled.
    

    ### Output Format:
    {format_instructions}
    """

    # Define format instructions
    format_instructions = """
    Please provide the output as a well-structured JSON object that adheres to the following format:

    {
    "gender": "<Specify the gender>",
    "Senior_Citizen": "<Yes/No>",
    "Is_Married": "<Yes/No>",
    "Dependents": "<Yes/No>",
    "tenure": "<Numeric value>",
    "Phone_Service": "<Yes/No>",
    "Dual": "<Yes/No/No phone service>",
    "Internet_Service": "<DSL/Fiber optic/No>",
    "Online_Security": "<Yes/No/No internet service>",
    "Online_Backup": "<Yes/No/No internet service>",
    "Device_Protection": "<Yes/No>",
    "Tech_Support": "<Yes/No>",
    "Streaming_TV": "<Yes/No>",
    "Streaming_Movies": "<Yes/No>",
    "Contract": "<Month-to-Month/One Year/Two Year>",
    "Paperless_Billing": "<Yes/No>",
    "Payment_Method": "<Credit Card/Bank Transfer/Electronic Check/Mailed Check>",
    "Monthly_Charges": "<Numeric value>",
    "Total_Charges": "<Numeric value>"
    }
    """


    # Create the PromptTemplate with history and text as input variables
    prompt = PromptTemplate(
        template=template,
        input_variables=["history", "text"],  # Include history and input
        partial_variables={"format_instructions": format_instructions}  # Static placeholder value
    )

    memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Initialize the LLM
    llm = Ollama(model="gemma2:latest")

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        input_key="text",  # Explicitly set the key for user input 
        )
    
    return conversation


def ML_model(data):
        """
        This function takes a dictionary representing customer data, preprocesses it, and uses a trained machine learning model to predict churn Parameters:
        output (dict): A dictionary containing customer data with the following keys: 
            - gender: A string representing the customer's gender. 
            - Senior_Citizen: A string representing whether the customer is a senior citizen. 
            - Is_Married: A string representing whether the customer is married. 
            - Dependents: A string representing whether the customer has dependents. 
            - tenure: A float representing the customer's tenure in months. 
            - Phone_Service: A string representing whether the customer has phone service. 
            - Dual: A string representing whether the customer has dual phone service. 
            - Internet_Service: A string representing the customer's internet service. 
            - Online_Security: A string representing whether the customer has online security. 
            - Online_Backup: A string representing whether the customer has online backup. 
            - Device_Protection: A string representing whether the customer has device protection. 
            - Tech_Support: A string representing whether the customer has tech support. 
            - Streaming_TV: A string representing whether the customer has streaming TV. 
            - Streaming_Movies: A string representing whether the customer has streaming movies. 
            - Contract: A string representing the customer's contract term. 
            - Paperless_Billing: A string representing whether the customer has paperless billing. 
            - Payment_Method: A string representing the customer's payment method. 
            - Monthly_Charges: A float representing the customer's monthly charges. 
            - Total_Charges: A float representing the customer's total charges. 
         
        Returns: 
        pandas.DataFrame: A DataFrame containing the preprocessed customer data and the predicted churn value. 
    """    
        try:
            df = pd.DataFrame([data])
            df.rename(columns={"Senior_Citizen": "Senior_Citizen "}, inplace=True)

            # Check for missing columns before accessing them.  Handle missing data gracefully.
            required_cols = ['Dependents', 'gender', 'Is_Married', 'Senior_Citizen ', 'tenure', 'Phone_Service', 
                                'Dual', 'Internet_Service', 'Online_Security', 'Online_Backup', 'Device_Protection',
                                'Tech_Support', 'Streaming_TV', 'Streaming_Movies', 'Contract', 'Paperless_Billing', 
                                'Payment_Method', 'Monthly_Charges', 'Total_Charges']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                st.error(f"Error: The following columns are missing from the LLM's output: {missing_cols}. Please check the LLM's response and ensure all necessary data is provided.")
                return None # Or handle the missing data appropriately, e.g., imputation

            # Map and encode categorical features; handle potential errors
            mapping = {
                'Dependents': {'Yes': 1, 'No': 0},
                'gender': {'Male': 1, 'Female': 0},
                'Is_Married': {'Yes': 1, 'No': 0},
                'Senior_Citizen ': {'Yes': 1, 'No': 0}
            }
            for col, mp in mapping.items():
                try:
                    df[col] = df[col].map(mp).fillna(0) # Fill NaN values with 0 if mapping fails
                except KeyError as e:
                    st.error(f"Error mapping values for column '{col}': {e}. Check the LLM's output for valid values.")
                    return None

            for column in df.select_dtypes(include=['object']).columns:
                try:
                    df[column] = LabelEncoder().fit_transform(df[column])
                except ValueError as e:
                    st.error(f"Error encoding column '{column}': {e}. Check for unexpected values in the LLM's output.")
                    return None


            scaler = StandardScaler()
            numerical_columns = ["tenure", 'Monthly_Charges', 'Total_Charges']
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

            ML_Model = load_model('My_Best_Pipeline')
            churn_prediction = ML_Model.predict(df.iloc[[-1]])
            df.loc[len(df) - 1, "Churn"] = churn_prediction[0]
            return df
        except Exception as e:
            st.exception(f"An error occurred in the ML model: {e}")
            return None


