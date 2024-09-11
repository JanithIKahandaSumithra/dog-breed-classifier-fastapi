# Dog Breed Classification FastAPI Application

This repository contains a FastAPI web application that serves a machine learning model for classifying dog breeds from images. The model is trained using TensorFlow, and the app provides an endpoint where users can upload images to receive breed predictions.

## Application Link

The FastAPI application is deployed on AWS and can be accessed at:

**[Dog Breed Prediction API](http://13.61.19.58:8000)**

- Use the `/` endpoint to check if the API is running.
- Use the `/Dog_Breed_Predict/` endpoint to upload an image and get the predicted breed.

## Project Structure

- `main.py`: FastAPI application code.
- `dog_breed_model.h5`: Pre-trained TensorFlow model for dog breed classification.
- `requirements.txt`: List of Python dependencies for the project.

## How to Run Locally

Follow the steps below to run the FastAPI app on your local machine:

### Prerequisites

- Python 3.7+
- Install dependencies listed in `requirements.txt` using pip.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/fastapi-dog-breed-classifier.git
    cd fastapi-dog-breed-classifier
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Start the FastAPI app:

    ```bash
    uvicorn main:app --reload
    ```

4. Access the application at `http://127.0.0.1:8000/` from your browser or API client (e.g., Postman).

## API Endpoints

- **Root Endpoint**:  
  `GET /`  
  Returns a welcome message to confirm the API is running.

- **Dog Breed Prediction**:  
  `POST /Dog_Breed_Predict/`  
  Accepts an image file and returns the predicted dog breed.

