from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from openai import OpenAI
from dotenv import load_dotenv

import os


# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-5.5")

# OpenAI client
client = OpenAI(api_key=api_key)


# FastAPI application
app = FastAPI()


# Serve static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Browser home page
@app.get("/")
def home():

    return FileResponse("static/index.html")


# AI API
@app.get("/ask")
def ask_ai(question: str):

    response = client.responses.create(
        model=model,
        input=question
    )

    return {
        "question": question,
        "answer": response.output_text
    }