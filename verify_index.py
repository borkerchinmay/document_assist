from sentence_transformers import SentenceTransformer
from src.indexing import build_index, search_documents
from src.ingestion import load_documents

docs= load_documents("data/documents")
for doc in docs:
    if doc["metadata"]["source"].endswith(".pdf"):
        print("\n", doc["text"])
    else:
        print("\nSOURCE:", doc["metadata"]["source"])
        print("PAGE:", doc["metadata"]["page"])
        print("TEXT:", doc["text"][:500])


model =  SentenceTransformer("all-MiniLM-L6-v2")

store = build_index("data/documents", model, chunk_size=60, overlap=15)
print("Total chunks:", len(store.documents))
print("FAISS vectors:", store.index.ntotal)

questions =["What is the deductible for accidental damage?",
           "Is damage to my own vehicle covered under the policy?",
           "Which required claim documents have not been submitted?"]

for q in questions:
    print(f"\nQUESTION: {q}")
    
    results = search_documents(q, store, model, top_k=3)
    print("Number of results:", len(results))

    for chunk, score in results:
        print("\nSource:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Similarity:", score)
        print("Text:", chunk["text"])

