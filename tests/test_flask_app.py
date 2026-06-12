import pytest
from flask_app.main import app
from flask_app import main as flask_main
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to our flask api" in response.data

def test_predict_with_timestamps_empty(client):
    response = client.post('/predict_with_timestamps', 
                          data=json.dumps({'comments': []}),
                          content_type='application/json')
    assert response.status_code == 400
    assert b"No comments provided" in response.data

def test_generate_chart_missing_data(client):
    response = client.post('/generate_chart', 
                          data=json.dumps({}), # missing sentiment_counts
                          content_type='application/json')
    assert response.status_code == 400


def test_predict_returns_500_when_model_load_fails(client, monkeypatch):
    flask_main.model = None
    flask_main.vectorizer = None
    flask_main.model_load_error = None

    def raise_load_error(*args, **kwargs):
        raise RuntimeError("cannot load model")

    monkeypatch.setattr(flask_main, "load_model_and_vectorizer", raise_load_error)

    response = client.post(
        '/predict',
        data=json.dumps({'comments': ["hello world"]}),
        content_type='application/json'
    )

    assert response.status_code == 500
    assert b"Prediction failed: Failed to load model and vectorizer" in response.data
