import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io

# Load your trained model
model = tf.keras.models.load_model("dog_breed_model.h5")

# Initialize the FastAPI app
app = FastAPI()

# Define the class names 
class_names = ["Beagle", "Boxer", "Bulldog", "Dachshund","German_Shepherd","Golden_Retriever","Labrador_Retriever","Poodle","Rottweiler","Yorkshire_Terrier"] 

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    # Resize the image to the dimensions expected by your model
    image = image.resize((150, 150))  # img_height and img_width used here
    image = np.array(image) / 255.0   # Normalize image data to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 150, 150, 3)
    return image

@app.get('/')
def index():
    return {'message': 'Dog Prediction API'}

# Define the prediction endpoint
@app.post("/Dog_Breed_Predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make the prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Return the predicted class name
    return {"predicted_breed": class_names[predicted_class]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
