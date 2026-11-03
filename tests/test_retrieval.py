from src.retrieval import chunk_text, retrieve

def test_chunk_text_creates_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("some text", chunk_size = 10, overlap=10)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        vectors = {"about cats": [1.0, 0.0], 
                  "about dogs": [0.0, 1.0],
                  "cats": [0.9, 0.1]}

        return np.array([vectors[t] for t in texts])

def test_retrieve_ranks_relevant_chunk_first():
    model = FakeEmbeddingModel()

    results = retrieve("cats", ["about cats", "about dogs"], model, top_k=1)

    assert results[0][0] == "about cats"
    
