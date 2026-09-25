from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "chroma_db"
DOCS = ROOT / "docs"

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("zepto_policies")
    except Exception:
        pass
    collection = client.create_collection(
        name="zepto_policies",
        metadata={"hnsw:space":"cosine"}
    )
    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        ids.append(path.stem)
        documents.append(text)
        metadatas.append({"source": path.name})
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} documents into {DB_DIR}")

if __name__ == "__main__":
    main()
