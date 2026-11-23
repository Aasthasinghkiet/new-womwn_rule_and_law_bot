
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_pipeline import ask_question, ask_llm_only
import traceback
import time

app = FastAPI(
    title="Women Law RAG Chatbot API",
    description="API for answering questions about Indian women's legal rights.",
    version="2.0"
)

# ---------------------------------------
# CORS CONFIGURATION
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# REQUEST MODEL
# ---------------------------------------
class Query(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )


# ---------------------------------------
# HOMEPAGE / HEALTH CHECK
# ---------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Women Law RAG API is running.",
        "endpoints": {
            "rag": "POST /ask",
            "normal_llm": "POST /ask_llm",
            "documentation": "/docs"
        }
    }


# ---------------------------------------
# RAG ENDPOINT
# ---------------------------------------
@app.post("/ask")
def ask_bot(data: Query):
    try:
        question = data.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        print("RAG Question received:", question)

        start_time = time.perf_counter()

        answer = ask_question(question)

        response_time = round(
            time.perf_counter() - start_time, 3
        )

        print("RAG answer generated successfully.")

        return {
            "method": "RAG",
            "question": question,
            "answer": answer,
            "response_time_seconds": response_time
        }

    except HTTPException:
        raise

    except Exception as e:
        print("========== RAG ERROR ==========")
        traceback.print_exc()
        print("================================")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate RAG response. Please try again."
        )


# ---------------------------------------
# NORMAL LLM ENDPOINT (WITHOUT RAG)
# ---------------------------------------
@app.post("/ask_llm")
def ask_normal_llm(data: Query):
    try:
        question = data.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        print("Normal LLM Question received:", question)

        start_time = time.perf_counter()

        answer = ask_llm_only(question)

        response_time = round(
            time.perf_counter() - start_time, 3
        )

        print("Normal LLM answer generated successfully.")

        return {
            "method": "Normal LLM",
            "question": question,
            "answer": answer,
            "response_time_seconds": response_time
        }

    except HTTPException:
        raise

    except Exception as e:
        print("========== NORMAL LLM ERROR ==========")
        traceback.print_exc()
        print("======================================")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate normal LLM response. Please try again."
        )
