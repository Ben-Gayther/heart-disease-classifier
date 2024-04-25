# heart-disease-classifier

## Installation

Run with python 3.12.2

```bash
pip install -r requirements.txt
```

## Docker Instructions

```bash
docker build -t heart_disease_image .
docker run -it -p 8000:8000 heart_disease_image
```

Will start the server on `http://localhost:8000/`

## API Prediction Endpoint

Send a GET request to `http://localhost:8000/predict` using the `test_app.py` script. This will return a JSON response like the following:

```json
{
    "predictions": [1.0, 0.0, 1.0],
    "probabilities": [0.85, 0.15, 0.75]
}
```

## Running streamlit app

```bash
streamlit run streamlit_app.py
```
