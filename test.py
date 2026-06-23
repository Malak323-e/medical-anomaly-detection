import requests
import os

# Test service preprocessing
def test_preprocessing(image_path):
    print("--- Test Preprocessing ---")
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://localhost:8002/preprocess",
            files={"file": f}
        )
    print(response.json())

# Test service inference
def test_inference(image_path):
    print("--- Test Inference ---")
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://localhost:8001/predict",
            files={"file": f}
        )
    print(response.json())

# Test service results
def test_save_result():
    print("--- Test Save Result ---")
    response = requests.post(
        "http://localhost:8003/save",
        json={
            "filename": "test_image.jpeg",
            "prediction": "ANOMALIE",
            "confidence": 87.5
        }
    )
    print(response.json())

def test_get_results():
    print("--- Test Get Results ---")
    response = requests.get("http://localhost:8003/results")
    print(response.json())

# Test health checks
def test_health():
    print("--- Health Checks ---")
    for port, name in [(8001, "inference"), (8002, "preprocessing"), (8003, "results")]:
        r = requests.get(f"http://localhost:{port}/health")
        print(f"{name}: {r.json()}")

if __name__ == "__main__":
    # Remplace par le chemin d'une vraie image du dataset
    IMAGE_PATH = "data/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    
    test_health()
    test_preprocessing(IMAGE_PATH)
    test_inference(IMAGE_PATH)
    test_save_result()
    test_get_results()