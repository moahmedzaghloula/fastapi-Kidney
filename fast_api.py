from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Load the trained model
model = joblib.load('kidney.pkl')
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=['*']
)



class Features(BaseModel):
    blood_pressure: int
    specific_gravity: float
    albumin: int
    sugar: int
    red_blood_cells: int
    blood_urea: int
    serum_creatinine: float
    sodium: float
    potassium: float
    hemoglobin: float
    white_blood_cell_count: int
    red_blood_cell_count: float
    hypertension: int

@app.post("/predict")
async def predict(features: Features):
    # Convert integer features to float
    features.blood_pressure = float(features.blood_pressure)
    features.blood_urea = float(features.blood_urea)
    features.white_blood_cell_count = float(features.white_blood_cell_count)
    features.hypertension = float(features.hypertension)
    features.albumin = float(features.albumin)
    features.sugar = float(features.sugar)
    features.red_blood_cells = float(features.red_blood_cells)
    
    # Convert features to list
    features_list = [ features.blood_pressure, features.specific_gravity, features.albumin,
                    features.sugar, features.red_blood_cells, features.blood_urea,
                    features.serum_creatinine, features.sodium, features.potassium, features.hemoglobin,
                    features.white_blood_cell_count, features.red_blood_cell_count,
                    features.hypertension]

    # Predicting
    prediction = model.predict([features_list])

    # Mapping prediction to result
    result = "Non-Chronic Kidney" if prediction == 0 else "Chronic Kidney"
    return {"prediction": result}


@app.get('/')
async def index():
    return {"message": "Welcome to the Kidney prediction API!"}


if __name__ == '__main__':
    uvicorn.run(app, port=5012)

