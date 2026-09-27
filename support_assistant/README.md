# Support Assistant

## Required baseline

The graded baseline is fully offline with respect to LLM calls.

Leave `MOCK_LLM` unset or set:

```bash
MOCK_LLM=1
```

Embeddings are generated locally with `all-MiniLM-L6-v2` and stored in ChromaDB.

## Ingestion

Run:

```bash
python support_assistant/ingest.py
```

This loads the eight required policy documents, creates one chunk per document, embeds each chunk, and persists the ChromaDB collection.

## API

Run:

```bash
uvicorn support_assistant.main:app --reload
```

Then POST:

```json
{
  "query": "How long do I have to report a damaged item?"
}
```

A policy query is routed to retrieval. A general query is routed directly to the canned general-answer node.

## Architecture

```text
8 policy documents
      |
      v
ingest.py / chunking
      |
      v
SentenceTransformer
all-MiniLM-L6-v2
      |
      v
ChromaDB collection: zepto_policies
      |
      v
POST /ask
      |
      v
LangGraph StateGraph
      |
      +--> classify_intent
             |
             +--> policy_question
             |       |
             |       v
             |   retrieve_and_answer
             |       |
             |       v
             |   top-3 ChromaDB chunks
             |
             +--> general_question
                     |
                     v
                 direct_answer
      |
      v
Pydantic response
answer / sources / confidence
```

In mock mode, intent classification and answer generation are deterministic. Retrieval still uses real local embeddings and ChromaDB. The optional `MOCK_LLM=0` path can call a real LLM, but it is not required for the graded baseline.

## Docker

From the repository root:

```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```


## Example calls to record after running locally

Policy query:

```json
{
  "query": "How long do I have to report a damaged item?"
}
```

Record the actual response printed by your running application here.

General query:

```json
{
  "query": "What is the capital of France?"
}
```

Record the actual response printed by your running application here.

Do not invent these responses; copy the raw JSON returned by your local application.
