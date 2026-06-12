import pytest
import json
from unittest.mock import patch, MagicMock

# Mock the model and vectorizer loading specifically in the main module
# to avoid breaking other libraries (like matplotlib) with global patches
with patch('mlflow.pyfunc.load_model', return_value=MagicMock()), \
     patch('pickle.load', return_value=MagicMock()), \
     patch('flask_app.main.load_model_and_vectorizer', return_value=(MagicMock(), MagicMock())):
    
    from flask_app.main import app

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
