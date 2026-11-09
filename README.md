# Enterprise Document Intelligence

A RAG-based document assistant that extracts information from PDF and TXT files and retrieves relevant passages using semantic search.

## Features
- PDF and TXT document ingestion.
- Text extraction and overlapping chunking.
- 384-dimensional embeddings using `all-MiniLM-L6-v2`.
- Semantic retrieval using cosine similarity and NumPy.
- FastAPI endpoints for document retrieval.
- Gemini API integration for answer generation.
- Retrieval evaluation using Precision@K, Recall@K, and latency.

## Tech Stack
Python · FastAPI · Sentence Transformers · NumPy · Gemini API · Pytest

## Evaluation Results

| Metric | Result |
|---|---:|
| Documents | 8 |
| Chunks | 17 |
| Evaluation Questions | 75 |
| Precision@3 | 46.22% |
| Recall@3 | 94.67% |
| Average Retrieval Latency | 7.64 ms |

*Evaluation uses source-document relevance labels. Latency excludes indexing and LLM generation.*

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ai_document_assistant.git
cd ai_document_assistant
```

**2. Install dependencies**
```bash
python -m venv .venv
pip install -r requirements.txt
```

**3. Configure API key**

Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key
```

**4. Run the application**
```bash
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for API documentation.

**5. Run tests and evaluation**
```bash
python -m pytest -q
python evaluate.py
```

## Future Improvements
- FAISS vector indexing.
- Hybrid search and reranking.
- Improved answer evaluation and source citations.

## Author
Chinmay Borker