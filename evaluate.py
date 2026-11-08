
import json
import time
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.retrieval import load_document, chunk_text


# Configuration
TOP_K = 3
DOCUMENT_DIR = Path("data/documents")
QUESTIONS_FILE = Path("evaluation/questions.json")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 1. Load and chunk all documents
all_chunks = []
chunk_sources = []

for file_path in DOCUMENT_DIR.iterdir():

    if file_path.suffix.lower() not in [".txt", ".pdf"]:
        continue

    text = load_document(file_path)

    chunks = chunk_text(
        text,
        chunk_size=120,
        overlap=30
    )

    all_chunks.extend(chunks)
    chunk_sources.extend([file_path.name] * len(chunks))


print("Documents loaded:", len(set(chunk_sources)))
print("Total chunks:", len(all_chunks))


# 2. Generate embeddings once
chunk_embeddings = model.encode(
    all_chunks,
    normalize_embeddings=True
)


# 3. Load evaluation questions
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)


precisions = []
recalls = []
latencies = []


# 4. Evaluate each question
for item in questions:

    question = item["question"]
    relevant_sources = set(item["relevant_sources"])

    # Find all relevant chunks in the corpus
    total_relevant_chunks = sum(
        source in relevant_sources
        for source in chunk_sources
    )

    if total_relevant_chunks == 0:
        print("Warning: No relevant source found:", question)
        continue

    # Measure retrieval latency
    start = time.perf_counter()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = chunk_embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][:TOP_K]

    end = time.perf_counter()

    latencies.append((end - start) * 1000)

    # Identify retrieved sources
    retrieved_sources = [
        chunk_sources[i] for i in indices
    ]

    # A retrieved chunk is relevant if its source is relevant
    relevant_retrieved = sum(
        source in relevant_sources
        for source in retrieved_sources
    )

    # Precision@K
    precision = relevant_retrieved / TOP_K

    # Recall@K
    recall = relevant_retrieved / total_relevant_chunks

    precisions.append(precision)
    recalls.append(min(recall, 1.0))


# 5. Display results
print("\n========== RAG EVALUATION ==========")

print("Questions evaluated:", len(precisions))
print("Top K:", TOP_K)

print(
    f"Average Precision@{TOP_K}: "
    f"{np.mean(precisions):.4f}"
)

print(
    f"Average Recall@{TOP_K}: "
    f"{np.mean(recalls):.4f}"
)

print(
    f"Average Retrieval Latency: "
    f"{np.mean(latencies):.2f} ms"
)

print(
    f"Maximum Retrieval Latency: "
    f"{np.max(latencies):.2f} ms"
)

print("====================================")
