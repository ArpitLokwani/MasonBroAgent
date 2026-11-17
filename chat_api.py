
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_app import (
    load_documents,
    create_chunks,
    create_embeddings,
    store_in_vector_db,
    retrieve_relevant_documents,
    generate_answer
)

app = FastAPI(title="MasonBro AI Chat API")

# Allow our website to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://masonbro.com",
        "https://www.masonbro.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"]
)

# Load RAG knowledge base when API starts
document = load_documents()
chunks = create_chunks(document)
embeddings = create_embeddings(chunks)
collection = store_in_vector_db(chunks, embeddings)


# Request format
class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "MasonBro AI API is running!"}


@app.post("/chat")
def chat(request: ChatRequest):

    try:
        relevant_documents = retrieve_relevant_documents(
            request.message,
            collection
        )

        answer = generate_answer(
            request.message,
            relevant_documents
        )

        return {
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate response"
        )
