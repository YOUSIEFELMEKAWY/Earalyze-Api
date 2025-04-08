from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
from pathlib import Path
import requests

# Function to download the model file
def download_model(model_url: str, save_path: Path):
    response = requests.get(model_url)
    with open(save_path, "wb") as f:
        f.write(response.content)

# Download the model if it's not already downloaded
model_path = Path("VGG19_classifier_model.h5")
if not model_path.exists():
    model_url = "https://drive.google.com/file/d/1iYiYiAbpD9hL3sdTJXXWLL5TNJgYRuol/view?usp=drive_link"  # Replace with your link to the model
    download_model(model_url, model_path)

# Load your trained model
model = load_model(str(model_path))
class_names = ["acute", "obtuse", "right"]

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def main():
    return """
    <html>
    <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: "Arial", sans-serif; }
            body { display: flex; justify-content: center; align-items: center; height: 100vh; background: linear-gradient(to right, #00c6ff, #0072ff); }
            .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2); text-align: center; width: 400px; position: relative; }
            h2 { color: #333; margin-bottom: 20px; }
            .upload-box { border: 2px dashed #0072ff; padding: 20px; border-radius: 10px; background: #f9f9f9; cursor: pointer; transition: 0.3s ease; }
            .upload-box:hover { background: #e3f2fd; }
            input[type="file"] { display: none; }
            .custom-file-upload { display: inline-block; padding: 10px 20px; background: #0072ff; color: white; border-radius: 5px; cursor: pointer; transition: 0.3s; }
            .custom-file-upload:hover { background: #005bb5; }
            input[type="submit"] { background: #28a745; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; margin-top: 15px; width: 100%; font-size: 16px; transition: 0.3s; }
            input[type="submit"]:hover { background: #218838; }
            .loading { display: none; margin-top: 15px; padding: 10px; background: #fff3cd; color: #856404; border-radius: 5px; border: 1px solid #ffeeba; text-align: center; }
            .result { margin-top: 20px; padding: 15px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; color: #0c5460; }
        </style>
        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
            }
            function updateFileName() {
                let fileInput = document.getElementById('fileInput');
                let fileNameDisplay = document.getElementById('fileName');
                if (fileInput.files.length > 0) {
                    fileNameDisplay.innerText = fileInput.files[0].name;
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h2>Upload an Image for Classification</h2>
            <form action="/predict/" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
                <label for="fileInput" class="upload-box">
                    <span id="fileName">Click to Choose a File</span>
                    <input type="file" name="file" id="fileInput" required onchange="updateFileName()">
                </label>
                <input type="submit" value="Upload and Predict">
            </form>
            <div id="loading" class="loading">
                <p>Processing... Please wait.</p>
            </div>
            <div id="result" class="result"></div>
        </div>
    </body>
</html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return {"filename": file.filename, "predicted_class": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
