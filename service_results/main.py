from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage en mémoire (simple pour le projet)
results_db = []

class Result(BaseModel):
    filename: str
    prediction: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/save")
def save_result(result: Result):
    entry = result.dict()
    entry["timestamp"] = datetime.now().isoformat()
    results_db.append(entry)
    return {"status": "saved", "id": len(results_db)}

@app.get("/results")
def get_results():
    return results_db