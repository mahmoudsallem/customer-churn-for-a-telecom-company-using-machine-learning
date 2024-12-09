import json
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List , Optional
from pydantic import BaseModel, ValidationError
from utails import parse, LLM_model , ML_model

# Initialize FastAPI app
app = FastAPI()

# Initialize parser and conversation model
parser = parse()
conversation = LLM_model()

# Global state
state = {
    "response_data": {},
    "missing_features": []
}


class InputData(BaseModel):
    user_input: str


class AdditionalData(BaseModel):
    missing_data: Dict[str, str]


@app.post("/get_data")
async def get_data(input_data: InputData):
    """
    Process user input, extract features, and determine missing fields.
    """
    response = conversation.run({"text": input_data.user_input})
    state["response_data"] = parser.parse(response)

    # Identify missing fields
    state["missing_features"] = [
        field for field, value in state["response_data"].items()
        if value in [None, "", "Unknown", "Not Specified"]
    ]

    return {
        "data": state["response_data"],
        "missing_features": state["missing_features"]
    }


@app.post("/add-missing")
async def add_missing_data(additional_data: AdditionalData):
    """
    Add missing data to the response and recheck for missing fields.
    """
    if not state["response_data"]:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please call /get_data first."
        )

    # Update response data with provided missing data
    for field, value in additional_data.missing_data.items():
        state["response_data"][field] = value

    # Recheck for missing fields
    state["missing_features"] = [
        field for field, value in state["response_data"].items()
        if value in [None, "", "Unknown", "Not Specified"]
    ]

    if state["missing_features"]:
        return {
            "message": "Some fields are still missing. Please provide the remaining data.",
            "missing_features": state["missing_features"]
        }

    return {
        "message": "All missing data has been provided. Thank you!",
        "data": state["response_data"]
    }


@app.post("/Prediction")
async def predict_with_ml():
    """
    Predict endpoint using the ML model if no missing data remains.
    """
    if not state["response_data"]:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please call /get_data first."
        )

    # Check for missing features
    if state["missing_features"]:
        return {
            "error": "There are missing fields in the data.",
            "missing_features": state["missing_features"]
        }

    # Pass valid data to the ML model
    prediction_result = ML_model(state["response_data"])

    if prediction_result is not None:
        return {"Prediction": prediction_result.to_dict(orient='records')}
    else:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while making the prediction."
        )
