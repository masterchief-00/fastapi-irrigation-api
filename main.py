from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import joblib
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

# Load the trained SVM model
model = joblib.load('./irrigation_model_svm.pkl')

# Load the list of training columns
with open('./training_columns.pkl', 'rb') as f:
    train_columns = pickle.load(f)

# Define the input data schema
class IrrigationData(BaseModel):
    soil_moisture: float
    temperature: float
    humidity: float
    time: datetime = datetime.now()
     # Add time attribute

class IrrigationResponse(BaseModel):
    irrigation_needed: int
    message: str

@app.post("/predict", response_model=IrrigationResponse)
def predict_irrigation(data: IrrigationData):
    try:
        # Extract features from the input data
        hour = pd.to_datetime(data.time).hour
        dayofweek = pd.to_datetime(data.time).dayofweek

        input_data = np.array([[
            data.soil_moisture, 
            data.temperature, 
            data.humidity
        ]])

        # Ensure the input data matches the training columns order
        if train_columns != ['Soil_Moisture', 'Temperature', 'Humidity', 'Hour', 'DayOfWeek']:
            raise HTTPException(status_code=500, detail="Training columns do not match expected order")

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Determine message based on prediction
        irrigation_needed = int(prediction[0])
        if irrigation_needed == 1:
            message = "Irrigation needed. Irrigation is going to take place."
        else:
            message = "No irrigation needed."

        # Return the prediction and message
        return IrrigationResponse(irrigation_needed=irrigation_needed, message=message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000,
                 reload=True)
