##from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware
#from pydantic import BaseModel
#from rag_pipeline import ask_question   # rag_pipeline.py must be in same folder

#app = FastAPI()

# ---------------------------------------
# CORS FIX (IMPORTANT FOR HTML FRONTEND)
# ---------------------------------------
#app.add_middleware(
    #CORSMiddleware,
    #allow_origins=["*"],        # allow all domains
    #allow_credentials=True,
    #allow_methods=["*"],
    #allow_headers=["*"],
#)

# ---------------------------------------
# Request model
# ---------------------------------------
#class Query(BaseModel):
    #question: str

# ---------------------------------------
# Homepage
# ---------------------------------------
#@app.get("/")
#def home():
    #return {"message": "Women Law RAG API is running. Use POST /ask to query."}

# ---------------------------------------
# Ask endpoint
# ---------------------------------------
#@app.post("/ask")
#def ask_bot(data: Query):
   # return {"answer": ask_question(data.question)}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import ask_question
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Women Law RAG API is running. Use POST /ask to query."
    }


@app.post("/ask")
def ask_bot(data: Query):
    try:
        print("Question received:", data.question)

        answer = ask_question(data.question)

        print("Answer generated successfully")

        return {"answer": answer}

    except Exception as e:
        print("========== RAG ERROR ==========")
        print(str(e))
        traceback.print_exc()
        print("================================")

        return {
            "error": "RAG backend error",
            "details": str(e)
        }