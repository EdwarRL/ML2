from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import joblib

app = FastAPI()

model = joblib.load('mnist_classification_model.plk')

@app.post("/Classifier/")
async def predict_number(image: UploadFile):
    new_size=(8,8)
    resized_image = image.resize(new_size)
    gs_image = resized_image.convert('L')
    matrix= np.array(gs_image)
    vector = matrix.flatten()
    predicted_number=model.predict(vector)

    return {"predicted_number": predicted_number}
