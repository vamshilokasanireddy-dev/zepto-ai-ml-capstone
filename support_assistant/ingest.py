from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB_DIR = ROOT / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(MODEL_NAME)

    documents = []
    ids = []
    metadatas = []

    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append(text)
        ids.append(path.stem)
        metadatas.append({"document_id": path.stem})

    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 documents, found {len(documents)}.")

    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    # Rebuild the collection contents deterministically.
    if ids:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Indexed {len(ids)} documents into {COLLECTION_NAME}.")
    print(f"Persistent database: {DB_DIR}")


if __name__ == "__main__":
    main()
