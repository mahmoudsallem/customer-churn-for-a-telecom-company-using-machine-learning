import streamlit as st
from langchain_ollama import OllamaLLM
from pycaret.classification import *
from langchain_core.prompts import ChatPromptTemplate
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import pandas as pd


# Set up Streamlit app title
st.title("ChatGPT-like Clone for Sequential Data Collection")

template = (
    "You are a chatbot designed to extract specific information from the following text content: {text}.\n\n"
    "Please adhere to the following instructions carefully:\n\n"
    "1. **Extract Information:** Only extract the following information, ensuring that the answers match the specified valid options:\n"
    "- **gender**: Valid answers are 'Male', 'Female'.\n"
    "- **Senior_Citizen**: Valid answers are 'Yes' or 'No'.\n"
    "- **Is_Married**: Valid answers are 'Yes' or 'No'.\n"
    "- **Dependents**: Valid answers are 'Yes' or 'No'.\n"
    "- **tenure**: Provide the tenure in months as a numeric value or convert it and show direct the result if it is year convert it to months and just show the number of months.\n"
    "- **Phone_Service**: Valid answers are 'Yes' or 'No'.\n"
    "- **Dual**: Valid answers are 'Yes', 'No', or 'No phone service'.\n"
    "- **Internet_Service**: Valid answers are 'DSL', 'Fiber optic', or 'No'.\n"
    "- **Online_Security**: Valid answers are 'Yes', 'No', or 'No internet service'.\n"
    "- **Online_Backup**: Valid answers are 'Yes', 'No', or 'No internet service'.\n"
    "- **Device_Protection**: Valid answers are 'Yes' or 'No'.\n"
    "- **Tech_Support**: Valid answers are 'Yes' or 'No'.\n"
    "- **Streaming_TV**: Valid answers are 'Yes' or 'No'.\n"    
    "- **Streaming_Movies**: Valid answers are 'Yes' or 'No'.\n"
    "- **Contract**: Valid answers are 'Month-to-Month', 'One Year', or 'Two Year'.\n"
    "- **Paperless_Billing**: Valid answers are 'Yes' or 'No'.\n"   
    "- **Payment_Method**: Valid answers include 'Credit Card', 'Bank Transfer', 'Electronic Check', or 'Mailed Check'.\n"
    "- **Monthly_Charges**: Provide a numeric value (e.g., '59.99').\n"
    "- **Total_Charges**: Provide a numeric value (e.g., '500.75').\n"
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response.\n\n"
    "3. **Prompt Clarification:** If the required information is not available in the text or does not match the valid options, ask for clarification or request the missing information.\n\n"
    "4. **Direct and Concise:** Your output should contain only the explicitly requested data, formatted cleanly and accurately, with no additional text.\n\n"
    "5. **invalid input:** If the user enters an invalid input, please respond with 'Invalid input. Please try again.'\n\n"
    "6. **Response:** Your response should be in the form of a string, with the requested information in the format: 'Key: Value'.\n\n"
    "7. **response** don't contain Here is the extracted information delet it"
)

# Initialize the OllamaLLM model
model = OllamaLLM(model="llama3")

# load the ML model
ML_model = load_model('My_Best_Pipeline')


df = pd.DataFrame(columns=[
                              "gender", "Senior_Citizen ", "Is_Married","Dependents","tenure", 
                              "Phone_Service", "Dual", "Internet_Service", "Online_Security", 
                              "Online_Backup", "Device_Protection", "Tech_Support", 
                              "Streaming_TV", "Streaming_Movies", "Contract", 
                              "Paperless_Billing", "Payment_Method", 
                              "Monthly_Charges", "Total_Charges"
                          ])


# Convert the template into a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template)

# Initialize session state for messages and responses
if "messages" not in st.session_state:
    st.session_state.messages = []
if "responses" not in st.session_state:
    st.session_state.responses = []

def encode_df(df):
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Is_Married'] = df['Is_Married'].map({'Yes': 1, 'No': 0})
    df['Senior_Citizen '] = df['Senior_Citizen '].map({'Yes': 1, 'No': 0})
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return df

def scale_df(df):
    scaler = StandardScaler()
    numerical_columns = ["tenure", 'Monthly_Charges', 'Total_Charges']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def ml_model (df):
    df = encode_df(df)
    df = scale_df(df)
    return ML_model.predict(df)


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Enter your query here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            # Format the prompt as a string
            formatted_prompt = prompt_template.format(text=user_input)
            
            # Get response from the model
            response = model.invoke(
                input=formatted_prompt,  # Pass the string as 'input'
                max_tokens=500,   # Set the maximum number of tokens to 500
                temperature=0.5  
            )

            # Display the response
            # st.markdown(response)
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save the extracted information in the responses list
            st.session_state.responses.append(response)
            response = re.sub("Here is the extracted information:\n\n", "", response)
            st.write(response)
            # Use regex to extract key-value pairs from the response
            pattern = r"(\w[\w\s]*):\s*(.+)"
            matches = re.findall(pattern, response)

            # Convert matches to a dictionary
            labels = {key.strip(): value.strip() for key, value in matches}
            df.loc[len(df)] = (labels.values())


            churn_prediction = ml_model(df.iloc[[-1]])
            df.loc[len(df) - 1, "Churn"] = churn_prediction[0]
            # Output the extracted labels
            # st.write(labels.keys())
            # st.write("\n")
            # st.write(labels.values())
            # st.write(ml_model(df.loc(len(df))))
            st.write(df)
        except Exception as e:
            # Handle any errors
            error_message = f"An error occurred: {str(e)}"
            st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# Option to display all responses
if st.button("Show Collected Information"):
    st.write(st.session_state.responses)

