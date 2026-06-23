# Medical Anomaly Detection

Malak El Filali/ Nour Khamalli

This project detects anomalies in chest X-ray images using a deep learning model.
It is built as a small microservice application with FastAPI and Docker.

The model classifies an uploaded chest X-ray as:

- `NORMAL`
- `ANOMALIE`

The project also includes preprocessing and result-saving services.

## Project Structure

```text
medical-anomaly-detection/
|-- data/
|   `-- chest_xray/
|-- service_inference/
|   |-- main.py
|   |-- model.py
|   |-- train.py
|   |-- chest_model.pth
|   `-- Dockerfile
|-- service_preprocessing/
|   |-- main.py
|   `-- Dockerfile
|-- service_results/
|   |-- main.py
|   `-- Dockerfile
|-- docker-compose.yml
|-- test.py
|-- index.html
`-- README.md
```

## What Each Part Does

`service_preprocessing`

Prepares the uploaded image before prediction. It resizes the image to `224x224`.

`service_inference`

Loads the trained ResNet18 model and predicts whether the chest X-ray is normal or anomalous.

`service_results`

Stores prediction results in memory. This is useful for testing and displaying previous predictions.

`index.html`

Frontend page for interacting with the services.

`test.py`

Simple Python script to test the APIs from the command line.

## Requirements

You need:

- Python 3.10 or newer
- Docker Desktop
- Git

If you want to run the services without Docker, you also need Python packages such as:

```bash
fastapi
uvicorn
pillow
python-multipart
torch
torchvision
requests
```

## Run With Docker

From the project root folder, run:

```bash
docker compose up --build
```

This starts three services:

| Service | URL | Purpose |
|---|---|---|
| Inference | `http://localhost:8001` | Predicts the class of an X-ray |
| Preprocessing | `http://localhost:8002` | Resizes/prepares images |
| Results | `http://localhost:8003` | Saves and returns results |

## Check If Services Are Running

Open these URLs in your browser:

```text
http://localhost:8001/health
http://localhost:8002/health
http://localhost:8003/health
```

Each one should return:

```json
{"status": "ok"}
```

## Test The API

After starting Docker, run:

```bash
python test.py
```

The script checks:

- health endpoints
- preprocessing endpoint
- prediction endpoint
- save result endpoint
- get results endpoint

Make sure the image path inside `test.py` exists:

```python
IMAGE_PATH = "data/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
```

If the image does not exist, replace it with another image from the dataset.

## API Endpoints

### Inference Service

Health check:

```http
GET http://localhost:8001/health
```

Predict image:

```http
POST http://localhost:8001/predict
```

The request must send an image file using form-data with the field name:

```text
file
```

Example response:

```json
{
  "prediction": "ANOMALIE",
  "confidence": 97.42,
  "probabilities": {
    "NORMAL": 2.58,
    "ANOMALIE": 97.42
  }
}
```

### Preprocessing Service

```http
POST http://localhost:8002/preprocess
```

Example response:

```json
{
  "status": "preprocessed",
  "size": "224x224",
  "format": "JPEG"
}
```

### Results Service

Save a result:

```http
POST http://localhost:8003/save
```

Example JSON body:

```json
{
  "filename": "test_image.jpeg",
  "prediction": "ANOMALIE",
  "confidence": 87.5
}
```

Get saved results:

```http
GET http://localhost:8003/results
```

## Train The Model

The training script is located here:

```text
service_inference/train.py
```

It uses a pretrained ResNet18 model and fine-tunes it on the chest X-ray dataset.

Expected dataset structure:

```text
data/chest_xray/chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- val/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

To train:

```bash
cd service_inference
python train.py
```

The script saves the best model as:

```text
service_inference/chest_model.pth
```

## Run Without Docker

You can also start each service manually.

Inference service:

```bash
cd service_inference
uvicorn main:app --host 0.0.0.0 --port 8001
```

Preprocessing service:

```bash
cd service_preprocessing
uvicorn main:app --host 0.0.0.0 --port 8002
```

Results service:

```bash
cd service_results
uvicorn main:app --host 0.0.0.0 --port 8003
```

## Important Notes

- This project is for learning and experimentation.
- It should not be used as a real medical diagnosis tool.
- The results service stores data only in memory, so saved results disappear when the service restarts.
- The trained model file `chest_model.pth` can be large.
- For future versions, large datasets and model files should usually be stored outside GitHub or with Git LFS.

## Author

Created by Malak.
