import requests
import streamlit as st
from requests.exceptions import RequestException

from app import PREDICTION_ENDPOINT


# Define a function to get a single prediction for serving the Streamlit app
def get_single_prediction(data: dict) -> int:
    try:
        response = requests.get(
            f"http://localhost:8000{PREDICTION_ENDPOINT}", json={"instances": [data]}
        )
        prediction = response.json()["predictions"][0]
        probability = response.json()["probabilities"][0]
        return prediction, probability
    except RequestException as e:
        raise Exception(f"Error during prediction: {e}")


def main():
    st.title("Heart Disease Prediction")

    age = st.slider("Age", min_value=0, max_value=120, value=50)
    chol = st.slider("Cholesterol", min_value=0, max_value=1000, value=180)
    resting_blood_pressure = st.slider(
        "Resting Blood Pressure", min_value=0, max_value=300, value=160
    )

    max_heart_rate = st.slider("Max Heart Rate", min_value=0, max_value=300, value=150)
    oldpeak = st.number_input(
        "Oldpeak", min_value=-99.99, max_value=6.0, value=0.8
    )  # just to test the -99.99 value

    sex = st.radio("Sex", options=[0, 1], index=0)
    fasting_blood_sugar = st.radio("Fasting Blood Sugar", options=[0, 1], index=0)
    exang = st.radio("Exang", options=[0, 1], index=0)

    chest_pain_type = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=2)
    resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2], index=0)
    slope = st.selectbox("Slope", options=[0, 1, 2], index=2)
    number_vessels_flourosopy = st.selectbox(
        "Number of Vessels Flourosopy", options=[0, 1, 2, 3, 4], index=2
    )
    thal = st.selectbox("Thal", options=[0, 1, 2, 3], index=2)

    if st.button("Predict"):
        data = {
            "age": age,
            "sex": sex,
            "chest pain type": chest_pain_type,
            "resting blood pressure": resting_blood_pressure,
            "chol": chol,
            "fasting blood sugar": fasting_blood_sugar,
            "resting ECG": resting_ecg,
            "max heart rate": max_heart_rate,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "number vessels flourosopy": number_vessels_flourosopy,
            "thal": thal,
        }

        try:
            prediction, probability = get_single_prediction(data)
            if prediction == 1:
                st.info(
                    f"The patient is predicted to have heart disease with a probability of {100*probability:.2f}%"
                )
            else:
                st.info(
                    f"The patient is predicted to not have heart disease with a probability of {100*(1-probability):.2f}%"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
