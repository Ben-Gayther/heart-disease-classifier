# heart-disease-classifier

## Code Overview

- `dev_work.ipynb` contains the initial data exploration and model training.
- `app.py` contains the FastAPI server and a prediction endpoint.
- `Dockerfile` contains the instructions to build the Docker image, and will start the server.
- `model.py` contains the model class and methods to load the model and make predictions. And also contains the data schemas defined with Pandera.
- `test_app.py` contains a script to test the prediction endpoint.
- `streamlit_app.py` contains a small streamlit app to interact with the model (via the server/endpoint).

## Installation

Run with python 3.12.2

```bash
pip install -r requirements.txt
```

## Running the FastAPI Server with Docker

```bash
docker build -t heart_disease_image .
docker run -it -p 8000:8000 heart_disease_image
```

Will start the server on `http://localhost:8000/`

## API Prediction Endpoint

Send a GET request to `http://localhost:8000/predict` using the `test_app.py` script. This will return a JSON response like the following:

```json
{
    "predictions": [1, 0, 1],
    "probabilities": [0.5677, 0.1234, 0.7789]
}
```

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```
