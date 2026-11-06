import numpy as np
from src.ingestion import load_documents, chunk_documents
from src.vector_store import VectorStore

def build_index(directory, model, chunk_size=120, overlap=30):
    documents = load_documents(directory)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    print("Documents loaded:", len(documents))
    print("Chunks generated:", len(chunks))
    for chunk in chunks:
        print(chunk["metadata"]["source"], chunk["chunk_id"])

    if not chunks:
        raise ValueError("No document chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, normalize_embeddings = True)
    print("Embedding shape:", embeddings.shape)

    store = VectorStore(embeddings.shape[1])
    store.add(embeddings, chunks)

    return store

def search_documents(question, store, model, top_k=3):
    query_embedding = model.encode([question], normalize_embeddings = True)[0]

    return store.search(query_embedding, top_k=top_k)

