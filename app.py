import re
import pandas as pd
from fastapi import FastAPI, HTTPException
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


# Initialize FastAPI app
app = FastAPI()

# Schema for structured response
class Info(BaseModel):
    gender: str = Field(description="Valid answers are 'Male', 'Female'", required=True)
    Senior_Citizen: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Is_Married: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Dependents: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    tenure: float = Field(description="Provide the tenure in months as a numeric value", required=True)
    Phone_Service: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Dual: str = Field(description="Valid answers are 'Yes', 'No', or 'No phone service'", required=True)
    Internet_Service: str = Field(description="Valid answers are 'DSL', 'Fiber optic', or 'No'", required=True)
    Online_Security: str = Field(description="Valid answers are 'Yes', 'No', or 'No internet service'", required=True)
    Online_Backup: str = Field(description="Valid answers are 'Yes', 'No', or 'No internet service'", required=True)
    Device_Protection: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Tech_Support: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Streaming_TV: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Streaming_Movies: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Contract: str = Field(description="Valid answers are 'Month-to-Month', 'One Year', or 'Two Year'", required=True)
    Paperless_Billing: str = Field(description="Valid answers are 'Yes' or 'No'", required=True)
    Payment_Method: str = Field(description="Valid answers include 'Credit Card', 'Bank Transfer', 'Electronic Check', or 'Mailed Check'", required=True)
    Monthly_Charges: float = Field(description="Provide a numeric value (e.g., '59.99')", required=True)
    Total_Charges: float = Field(description="Provide a numeric value (e.g., '500.75')", required=True)

# Create a Pydantic output parser
parser = JsonOutputParser(pydantic_object=Info)


format_instructions = parser.get_format_instructions()

# Initialize the LLM
llm  = Ollama(model="llama3")

# Define the template for the prompt
template = "You are a chatbot. Extract the features from the text and respond in the following JSON format.  The JSON should be a single object, not an array of objects:\n{text}\n{format_instructions}"

prompt = PromptTemplate(
    template=template,
    input_variables=['text'],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt|llm


# load the ML model
ML_model = load_model('My_Best_Pipeline')


# Convert the template into a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template)


# Define encoding function
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
    return ML_model.predict(df)

class InputData(BaseModel):
    user_input: str

# FastAPI endpoint for processing input and generating response
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        
        # Invoke the LLM with the formatted prompt
        response = chain.invoke([HumanMessage(content=input_data.user_input)])
        output = parser.invoke(response)
        
        # Parse the response and process it

        df = pd.DataFrame([output])

        df.rename(columns={"Senior_Citizen": "Senior_Citizen "}, inplace=True)

        churn_prediction = ml_model(df.iloc[[-1]])
        df.loc[len(df) - 1, "Churn"] = churn_prediction[0]

        # Return the dataframe as a response
        return {"result": df.to_dict(orient="records")}
        # return df

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

