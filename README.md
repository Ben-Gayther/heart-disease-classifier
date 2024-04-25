# heart-disease-classifier

## Code Overview

- `notebooks/dev_work.ipynb` contains the initial data exploration and model training.
- `heart-disease-classifier/app.py` contains the FastAPI server and a prediction endpoint.
- `heart-disease-classifier/model.py` contains the model class and methods to load the model and make predictions. And also contains the data schemas defined with Pandera.
- `heart-disease-classifier/test_app.py` contains a script to test the prediction endpoint.
- `heart-disease-classifier/streamlit_app.py` contains a small streamlit app to interact with the model (via the server/endpoint).
- `Dockerfile` contains the instructions to build the Docker image, and will start the server.

The model itself is stored as a pickle file in the `models` directory.

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

Running the image will start the server locally on `http://localhost:8000/`

## API Prediction Endpoint

Whilst the server is running, send a GET request to the endpoint `http://localhost:8000/predict` using the `test_app.py` script, by running the following command:

```bash
python heart-disease-classifier/test_app.py
```

This will return a JSON response with the following fields, for example:

```json
{
    "predictions": [1, 0, 1],
    "probabilities": [0.5677, 0.1234, 0.7789]
}
```

## Running the Streamlit App

To run the streamlit app, run the following command:

```bash
streamlit run heart-disease-classifier/streamlit_app.py
```
