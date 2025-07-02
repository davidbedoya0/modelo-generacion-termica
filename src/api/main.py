from fastapi import FastAPI
from src.models.prediction import ModelPrediction


app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def viewPredict():
    """
    Endpoint to handle prediction requests.
    This function should be extended to include the actual prediction logic.
    """
    # Placeholder for prediction logic
    Model_prediction = ModelPrediction(
        model_name="example_model",
        prediction_value=42.0,
        confidence=0.95
    )
    prediction = Model_prediction.predict()
    prediction = prediction.dict()  # Convert to dictionary for response
    response = {
        "model_name": Model_prediction.model_name,
        "prediction_value": prediction['prediction_value'],
        "confidence": prediction['confidence']
    }
    
    return response 
