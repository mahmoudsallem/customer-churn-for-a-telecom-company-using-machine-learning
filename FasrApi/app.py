# import json
# import uvicorn
# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse
# from typing import Dict, List , Optional
# from pydantic import BaseModel
# from utails import parse, LLM_model

# # Assuming parse and LLM_model are defined elsewhere
# # from some_module import parse, LLM_model

# app = FastAPI()

# # Initialize parser and conversation model
# parser = parse()
# conversation = LLM_model()


# class InputData(BaseModel):
#     user_input: str

# @app.post("/predict")
# async def predict(input_data: InputData):
    
#    global missing_features , response
   
#    response = conversation.run({"text":input_data.user_input})

#    response_data = parser.parse(response)  # Assuming parse method is called parse

#    missing_features = [field for field in response_data if response_data[field] in [None, "", "Unknown", "Not Specified"]]
    

#    return {"data" :response_data , "Missing": missing_features}

# @app.post("/add-missing")
# async def add_missing_data(additional_data: Dict[str, str]):
#     """
#     Accepts a JSON object with key-value pairs for missing data and updates the response.
#     """
#     global session_data

#     # Ensure `additional_data` is treated as a JSON object
#     if not isinstance(additional_data, dict):
#         return {"error": "Invalid input. Please provide a JSON object."}

#     # Add the missing data provided by the user
#     session_data["response_data"].update(additional_data)

#     # Recheck for remaining missing fields
#     session_data["missing_features"] = [
#         field for field in session_data["missing_features"]
#         if field not in additional_data or additional_data[field] in [None, "", "Unknown", "Not Specified"]
#     ]

#     # Check for still-missing fields
#     if session_data["missing_features"]:
#         return {
#             "message": "Some fields are still missing. Please provide the remaining data.",
#             "missing_features": session_data["missing_features"]
#         }

#     # If all missing data has been provided
#     return {
#         "message": "All missing data has been provided. Thank you!",
#         "data": session_data["response_data"]
#     }
# #______________________________________________________________________________________________________________________


from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel, ValidationError
from utails import parse, LLM_model , ML_model

# Initialize FastAPI app
app = FastAPI()

# Initialize parser and conversation model
parser = parse()
conversation = LLM_model()

# In-memory state management
# state = {
#     "response_data": {},
#     "missing_features": []
# }
global response_data , missing_features
class InputData(BaseModel):
    user_input: str


class AdditionalData(BaseModel):
    missing_data: Dict[str, str]


@app.post("/get_data")
async def predict(input_data: InputData):
    """
    Predict endpoint to process user input, extract features, and determine missing fields.
    """
    response = conversation.run({"text": input_data.user_input})
    response_data = parser.parse(response)

    # Identify missing fields
    missing_features = [
        field for field, value in response_data.items()
        if value in [None, "", "Unknown", "Not Specified"]
    ]

    # Update state
   #  state["response_data"] = response_data
   #  state["missing_features"] = missing_features

    return {"data": response_data, "missing_features": missing_features}


@app.post("/add-missing")
async def add_missing_data(additional_data: AdditionalData):
    """
    Add missing data to the previously processed response and recheck for missing fields.
    """
    if not response_data:
        raise HTTPException(
            status_code=400,
            detail="No prediction data available. Please call /predict first."
        )

    # Update response data with provided missing data
    for field, value in additional_data.missing_data.items():
        response_data[field] = value

    # Recheck for missing fields
    missing_features = [
        field for field, value in response_data.items()
        if value in [None, "", "Unknown", "Not Specified"]
    ]

    # Return response
    if missing_features:
        return {
            "message": "Some fields are still missing. Please provide the remaining data.",
            "missing_features": missing_features
        }

    return {
        "message": "All missing data has been provided. Thank you!",
        "data": response_data
    }

@app.post("/Prediction")
async def predict_with_ml():
    if response_data:
      # return {"Prediction": ML_model(state["response_data"])}
      return {"data" : response_data}
   #  else:
   #    return {"error": "No prediction data available. Please call /predict first."}
   # else:
   #    return {"data": "Please pass all data."}