from fastapi import FastAPI
from pydantic import BaseModel

from .graph import SupportResponse, ask

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=SupportResponse)
def ask_endpoint(request: AskRequest) -> SupportResponse:
    return ask(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
