from pathlib import Path
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
EXCLUDED_FILES = {"README.txt", "sample.txt"}
def load_documents(directory):

    directory = Path(directory)


    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(directory)

    documents = []

    for file_path in sorted(directory.iterdir()):
        if file_path.name in EXCLUDED_FILES:
            continue
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))

            for page_num, page in enumerate(reader.pages, start =1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append({"text": text,
                                      "metadata": {"source": file_path.name,
                                                   "page": page_num}})


        elif file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")

            if text.strip():
                documents.append({"text": text,
                                  "metadata": {"source": file_path.name,
                                                "page": None}})

    return documents

def chunk_documents(documents, chunk_size = 120, overlap = 30):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        words= document["text"].split()
        metadata = document["metadata"]
        source = metadata["source"]
        page = metadata["page"]

        for chunk_index, start in enumerate(range(0,len(words), step)):

            chunk_text = " ".join(words[start:start + chunk_size])

            if not chunk_text:
                continue

            page_label = page if page is not None else "na"

            chunks.append({"chunk_id":(f"{source}_p{page_label}_chunk_{chunk_index}"),
                           "text": chunk_text,
                           "metadata": metadata.copy()})

    return chunks
        