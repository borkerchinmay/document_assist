from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.indexing import build_index, search_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS index
    store = build_index(
        "data/documents",
        model,
        chunk_size=120,
        overlap=30
    )

    app.state.embedding_model = model
    app.state.vector_store = store

    yield


app = FastAPI(
    title="Enterprise Document Intelligence",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    model = app.state.embedding_model
    store = app.state.vector_store

    results = search_documents(
        request.question,
        store,
        model,
        top_k=3
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant document passages found."
        )

    return {
        "question": request.question,
        "results": [
            {
                "context": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "similarity": score
            }
            for chunk, score in results
        ]
    }