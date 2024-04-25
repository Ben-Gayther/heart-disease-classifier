import requests

sample_data = {
    "age": 65,
    "sex": 0,
    "chest pain type": 2,
    "resting blood pressure": 160,
    "chol": 360,
    "fasting blood sugar": 0,
    "resting ECG": 0,
    "max heart rate": 151.0,
    "exang": 0,
    "oldpeak": 0.8,
    "slope": 2,
    "number vessels flourosopy": 0,
    "thal": 2,
    # "target": 1.0, # won't be present in the request
}

# test single instance
response = requests.get(
    "http://localhost:8000/predict", json={"instances": [sample_data]}
)

print(f"Status code: {response.status_code}")
assert response.status_code == 200
assert len(response.json()["predictions"]) == 1
assert "probabilities" in response.json()
assert "predictions" in response.json()

# duplicate the sample data to test multiple instances
N = 1_000

response = requests.get(
    "http://localhost:8000/predict",
    json={"instances": [sample_data for _ in range(N)]},
)

print(f"Status code: {response.status_code}")
assert response.status_code == 200
assert len(response.json()["predictions"]) == N
assert "probabilities" in response.json()
assert "predictions" in response.json()


# Can make these into proper tests using pytest...
# e.g. test_single_instance_prediction

print("All assertions passed.")
