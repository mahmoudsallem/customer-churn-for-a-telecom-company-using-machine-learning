# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecom company using machine learning develop chatbot to extract data from the text and classify it into churn or not.

## Dataset

The project uses the Telco Customer Churn dataset, which is publicly available. You can find the dataset on Kaggle. The dataset contains information about customers, including demographics, services subscribed to, and churn status.

## Features

- **Chatbot Interface**: A user-friendly interface for interacting with the chatbot.
- **Data Extraction**: Extracts specific information from user inputs using a predefined template.
- **Machine Learning Prediction**: Uses a pre-trained machine learning model to predict outcomes based on the extracted data.
- **Data Encoding and Scaling**: Preprocesses data using encoding and scaling techniques to prepare it for machine learning predictions.
- **Session Management**: Maintains chat history and responses using Streamlit's session state.


## Libraries

The following libraries are used in this project:

- **pandas**
- **numpy**
- **seaborn**
- **scikit-learn**
- **imblearn**
- **dataprep**
- **pycaret**
- **fastapi**
- **streamlit** 

## Preprocessing

The following preprocessing steps are performed:

- Drop the customerID column
- Map categorical features to numerical values
- Handle imbalanced classes using oversampling
- Split the data into training and testing sets
- Scale numerical features using StandardScaler

## Model Training and Evaluation

Several machine learning models are trained and evaluated, including:

- **Logistic Regression**
- **K-Nearest Neighbors**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **Support Vector Machine**

The models are compared using various metrics, such as accuracy, precision, recall, F1-score, and Fbeta-score. PyCaret is also used for automated machine learning, including model selection and hyperparameter tuning.

## Results

The results of the model training and evaluation are presented in the notebook. The best-performing model is selected based on the evaluation metrics.

## Usage

To run this project, you will need to:

- Install the required libraries
- Download the Telco Customer Churn dataset
- Upload the dataset to your Google Colab environment
- Run the code in the notebook


## Project Structure

- `main.py`: The main application file containing the Streamlit app and chatbot logic.
- `app.py`: The fastapi app file containing the user endpoint of the app.
- `My_Best_Pipeline`: The pre-trained machine learning model used for predictions.
- `README.md`: This file, providing an overview and usage instructions for the project.
- `requirements.txt`: A file containing the required libraries and their versions.



## Installation

1. **To Use Streamlit**:
   ```bash
    git clone <repository-url>
   
    cd <repository-directory>

    pip install -r requirements.txt 
    
    streamlit run main.py 


2. **To Use FastApi**:
   ```bash
    git clone <repository-url>
   
    cd <repository-directory>

    pip install -r requirements.txt

    uvicorn app:app --reload
