from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from model import MODEL_PATH, HeartDiseaseClassifier

PREDICTION_ENDPOINT = "/predict"


# Define the response schema
class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]

    # Validate the predictions and probabilities fields (outputs of the model)
    @field_validator("predictions")
    def validate_predictions(cls, predictions):
        if set(predictions) not in [{0}, {1}]:
            raise ValueError("Predictions must be either 0 or 1")
        return predictions

    @field_validator("probabilities")
    def validate_probabilities(cls, probabilities):
        if max(probabilities) > 1 or min(probabilities) < 0:
            raise ValueError("Probabilities must be between 0 and 1")
        return probabilities


router = FastAPI()

model = HeartDiseaseClassifier(MODEL_PATH)


@router.get(PREDICTION_ENDPOINT, response_model=PredictionResponse)
async def predict(request: Request):
    body = await request.json()
    try:
        # instances is a list of dictionaries with feature values
        instances = body["instances"]
        df = pd.DataFrame(instances)
        preds = model.predict(df)
        probs = model.predict_proba(df)[
            :, 1
        ]  # return only the probabilities of the positive class
        return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
    except Exception as e:
        print(f"Request failed with error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:router", host="localhost", port=8000, log_level="info", reload=True
    )
