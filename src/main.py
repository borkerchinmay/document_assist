from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.retrieval import load_document, chunk_text, retrieve

app = FastAPI(title="AI Document Assistant")

DOCUMENT_PATH = Path("data/documents/sample.txt")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

text = load_document(DOCUMENT_PATH)
chunks = chunk_text(text, chunk_size=35, overlap=8)
chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True)


class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results = retrieve(request.question, chunks, embedding_model, top_k=1, chunk_embeddings=chunk_embeddings)

    if not results:
        raise HTTPException(status_code=404,
                            detail="No relevant document passages found.")

    return {"question": request.question,
            "context": results[0][0],
            "similarity": results[0][1]}