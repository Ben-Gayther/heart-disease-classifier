import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from pandera import DataFrameModel, Field, check_types
from pandera.typing import Category, DataFrame, Series
from xgboost import XGBClassifier

MODEL_PATH = "model.pkl"


class InputDataSchema(DataFrameModel):
    # Can adjust checks and data types as needed

    age: Series[int] = Field(ge=0)  # greater than or equal to 0
    cholesterol: Series[int] = Field(ge=0, alias="chol")
    restingBloodPressure: Series[int] = Field(ge=0, alias="resting blood pressure")

    oldPeak: Series[float] = Field(
        alias="oldpeak"
    )  # could put ge=0 here instead to trigger a schema error earlier
    maxHeartRate: Series[float] = Field(ge=0, nullable=True, alias="max heart rate")

    sex: Series[Category] = Field(isin=[0, 1])
    chestPainType: Series[Category] = Field(isin=[0, 1, 2, 3], alias="chest pain type")
    fastingBloodSugar: Series[Category] = Field(
        isin=[0, 1], alias="fasting blood sugar"
    )
    exang: Series[Category] = Field(isin=[0, 1])
    slope: Series[Category] = Field(isin=[0, 1, 2])
    thal: Series[Category] = Field(isin=[0, 1, 2, 3])
    numberVesselsFlourosopy: Series[Category] = Field(
        isin=[0, 1, 2, 3, 4],
        alias="number vessels flourosopy",
    )
    restingECG: Series[Category] = Field(isin=[0, 1, 2], alias="resting ECG")

    target: Optional[Series[Category]] = Field(isin=[0, 1])

    class Config:
        coerce = True  # will coerce the data types to the specified types


class TrainingDataSchema(InputDataSchema):
    # Can add any new engineered features / any additional checks here

    oldPeak: Series[float] = Field(
        ge=0, nullable=True, alias="oldpeak"
    )  # as we are replacing -99.99 values with NaN


class HeartDiseaseClassifier:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)

        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            # initialise model with parameters used in initial training
            self.model = XGBClassifier(random_state=42, enable_categorical=True)

    @staticmethod
    @check_types
    def prepare_data(df: DataFrame[InputDataSchema]) -> DataFrame[TrainingDataSchema]:
        df["oldpeak"] = df["oldpeak"].replace(-99.99, np.nan)
        return df

    @check_types
    def predict(self, X: DataFrame[InputDataSchema]) -> np.ndarray:
        X_transformed = self.prepare_data(X)
        return self.model.predict(X_transformed)

    @check_types
    def predict_proba(self, X: DataFrame[InputDataSchema]) -> np.ndarray:
        X_transformed = self.prepare_data(X)
        return self.model.predict_proba(X_transformed)

    @check_types
    def train(self, X: DataFrame[InputDataSchema], y: Series[Optional[int]]):
        X_transformed = self.prepare_data(X)
        self.model.fit(X_transformed, y)

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
