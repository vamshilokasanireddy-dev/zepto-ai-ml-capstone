# Support Assistant

## Run

From this folder:

```bash
pip install -r ../requirements.txt
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

The default `MOCK_LLM` mode is deterministic and requires no API key.

## Example 1 — policy retrieval

Request:

```json
{"query":"What is the delivery fee below INR 149?"}
```

Expected shape:

```json
{"answer":"Based on the retrieved context: ...","sources":["doc_01"],"confidence":1.0}
```

## Example 2 — general question

```json
{"query":"What is the capital of India?"}
```

Expected shape:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Architecture

`docs/*.txt` → `ingest.py` chunk/document loading → `all-MiniLM-L6-v2` embeddings → ChromaDB collection `zepto_policies` → `classify_intent` → `retrieve_and_answer` or `direct_answer` → Pydantic `AskResponse` → FastAPI `/ask`.

In default/mock mode classification uses the required keyword heuristic and generation is deterministic. Retrieval still uses real local embeddings and ChromaDB. The optional real-LLM branch is activated only with `MOCK_LLM=0`.
