# Importing the libraries

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
from typing import Dict

# Define the model for prediction request
class PredictionRequest(BaseModel):
    user_id: str
    features: Dict[str, float]

# Define the machine learning service
class MLService:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.setup_monitoring()

        # Load or train the model
        self.model = self.load_or_train_model()

    def load_or_train_model(self):
        # Load the dataset and train the model
        try:
            model = joblib.load("random_forest_model.pkl")
            print("Model loaded from random_forest_model.pkl")
        except FileNotFoundError:
            dataset = pd.read_csv('Position_Salaries.csv')
            X = dataset.iloc[:, 1:-1].values
            y = dataset.iloc[:, -1].values

            # Train the model
            model = RandomForestRegressor(n_estimators=10, random_state=0)
            model.fit(X, y)
            joblib.dump(model, "random_forest_model.pkl")
            print("Model saved as random_forest_model.pkl")
        return model

    def setup_routes(self):
        @self.app.get("/")
        def read_root():
            return {"message": "Welcome to the Random Forest Salary Prediction API"}

        @self.app.post("/predict/")
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            try:
                validated_features = self.validate_input(request.features)

                prediction = self.model.predict([list(validated_features.values())])[0]

                background_tasks.add_task(self.drift_detector, validated_features)

                logging.info("Prediction made", extra={"user_id": request.user_id, "prediction": prediction})

                return {"predicted_salary": prediction}
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")

    def setup_monitoring(self):
        logging.basicConfig(level=logging.INFO)

    def validate_input(self, features: Dict[str, float]):
        return features

    async def drift_detector(self, features: Dict[str, float]):
        pass

# Initialize the MLService instance and FastAPI app
ml_service = MLService()
app = ml_service.app
