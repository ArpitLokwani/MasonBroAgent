
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------
# 1. Load configuration
# --------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("OPENAI_MODEL", "gpt-5.5")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

# Embedding model converts text into vectors
embedding_model = "text-embedding-3-small"


# --------------------------------
# 2. Load MasonBro documents
# --------------------------------

def load_documents():

    file_path = Path("data/masonbro_faq.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    print("MasonBro document loaded successfully!")

    return text


# --------------------------------
# 3. Split document into chunks
# --------------------------------

def create_chunks(text):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) > 500:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"Created {len(chunks)} chunks")

    return chunks


# --------------------------------
# 4. Generate embeddings
# --------------------------------

def create_embeddings(chunks):

    response = client.embeddings.create(
        model=embedding_model,
        input=chunks
    )

    embeddings = [
        item.embedding for item in response.data
    ]

    print("Embeddings generated successfully!")

    return embeddings


# --------------------------------
# 5. Store information in ChromaDB
# --------------------------------

def store_in_vector_db(chunks, embeddings):

    db_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = db_client.get_or_create_collection(
        name="masonbro_knowledge"
    )

    # Remove older MasonBro FAQ chunks before re-indexing
    if collection.count() > 0:
        collection.delete(
            where={"source": "masonbro_faq.txt"}
        )

    ids = [
        f"masonbro_{i}" for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": "masonbro_faq.txt"}
            for _ in chunks
        ]
    )

    print("Data stored in ChromaDB!")

    return collection


# --------------------------------
# 6. Retrieve relevant information
# --------------------------------

def retrieve_relevant_documents(question, collection):

    question_embedding = client.embeddings.create(
        model=embedding_model,
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, collection.count())
    )

    documents = results["documents"][0]

    print("\nRelevant information retrieved:")

    for document in documents:
        print("-", document)

    return documents


# --------------------------------
# 7. Generate answer using LLM
# --------------------------------

def generate_answer(question, documents):

    context = "\n\n".join(documents)

    prompt = f"""
    You are the MasonBro customer support assistant.

    Answer the question using only the information
    provided in the context.

    If the answer is not available in the context,
    say that you do not have that information.

    Context:
    {context}

    Customer question:
    {question}
    """

    response = client.responses.create(
        model=llm_model,
        input=prompt
    )

    return response.output_text


# --------------------------------
# 8. Main application
# --------------------------------

if __name__ == "__main__":

    document = load_documents()

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    collection = store_in_vector_db(
        chunks,
        embeddings
    )

    question = input("\nAsk MasonBro AI: ")

    relevant_documents = retrieve_relevant_documents(
        question,
        collection
    )

    answer = generate_answer(
        question,
        relevant_documents
    )

    print("\nMasonBro AI Answer:")
    print(answer)
