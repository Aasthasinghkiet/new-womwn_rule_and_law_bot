
import os
import re
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from openai import OpenAI


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
TEXT_DIR = BASE_DIR / "data" / "cleaned_texts"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTOR_DIR = BASE_DIR / "embeddings" / "vectors"

COLLECTION_NAME = "women_law_acts_v2"

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 6

# Load environment variables
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to your .env file."
    )

client = OpenAI(api_key=API_KEY)

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY
)


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def clean_text(text):
    """Clean extracted PDF text without removing useful content."""

    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_section(text):
    """
    Detect common Indian legal section references.

    This is metadata assistance, not a guarantee that
    every detected number is a statutory section.
    """

    pattern = re.compile(
        r"\b(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?)",
        re.IGNORECASE
    )

    matches = pattern.findall(text)

    return list(dict.fromkeys(matches))


def get_vector_db():
    """Load the existing ChromaDB collection."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embedding_model
    )


# =====================================================
# 3. PDF TEXT EXTRACTION
# =====================================================

def extract_text():

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:", PDF_DIR)
        return

    for pdf_path in pdf_files:

        print("Extracting:", pdf_path.name)

        try:

            reader = PdfReader(str(pdf_path))

            extracted_pages = []

            for page_number, page in enumerate(
                reader.pages, start=1
            ):

                page_text = page.extract_text() or ""

                page_text = clean_text(page_text)

                if page_text:

                    extracted_pages.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "content": page_text
                    })

            output_file = TEXT_DIR / (
                pdf_path.stem + ".json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    extracted_pages,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"Successfully extracted {len(extracted_pages)} pages."
            )

        except Exception as e:

            print(
                f"Error extracting {pdf_path.name}: {e}"
            )

    print("\nPDF extraction completed.")


# =====================================================
# 4. TEXT CHUNKING
# =====================================================

def chunk_text():

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            " ",
            ""
        ]
    )

    total_chunks = 0

    for file_path in TEXT_DIR.glob("*.json"):

        print("Chunking:", file_path.name)

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                pages = json.load(f)

            documents = []

            for page_data in pages:

                page_content = page_data["content"]

                chunks = splitter.split_text(
                    page_content
                )

                for index, chunk in enumerate(chunks):

                    documents.append({

                        "source": page_data["source"],

                        "page": page_data["page"],

                        "chunk_id": index,

                        "sections": extract_section(chunk),

                        "content": chunk

                    })

            output_file = CHUNKS_DIR / (
                file_path.stem + ".json"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    documents,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            total_chunks += len(documents)

            print(
                f"Created {len(documents)} chunks."
            )

        except Exception as e:

            print(
                f"Error chunking {file_path.name}: {e}"
            )

    print(
        f"\nTotal chunks created: {total_chunks}"
    )


# =====================================================
# 5. GENERATE EMBEDDINGS
# =====================================================

def create_embeddings():

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in CHUNKS_DIR.glob("*.json"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        for item in chunks:

            documents.append(

                Document(

                    page_content=item["content"],

                    metadata={

                        "source": item["source"],

                        "page": int(item["page"]),

                        "chunk_id": int(item["chunk_id"]),

                        "sections": ", ".join(
                            item.get("sections", [])
                        )

                    }

                )

            )

    if not documents:

        raise ValueError(
            "No chunks found. Run extract_text() and chunk_text() first."
        )

    print(
        f"Total chunks to embed: {len(documents)}"
    )

    # Use a dedicated collection.
    # Delete only this collection before rebuilding.
    # Other Chroma collections remain untouched.

    try:

        existing_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DIR),
            embedding_function=embedding_model
        )

        existing_db.delete_collection()

    except Exception:

        pass

    # Create a fresh vector database

    Chroma.from_documents(

        documents=documents,

        embedding=embedding_model,

        collection_name=COLLECTION_NAME,

        persist_directory=str(VECTOR_DIR),

        batch_size=100

    )

    print(
        "\nEmbeddings successfully generated!"
    )

    print(
        "Vector database:", VECTOR_DIR
    )


# =====================================================
# 6. RETRIEVE RELEVANT LEGAL CONTEXT
# =====================================================

def retrieve_context(query, k=TOP_K):

    db = get_vector_db()

    results = db.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    retrieved_documents = []

    for document, score in results:

        # Discard very weak matches.
        # Tune this threshold after testing real queries.
        if score < 0.15:
            continue

        retrieved_documents.append({

            "content": document.page_content,

            "source": document.metadata.get(
                "source", "Unknown document"
            ),

            "page": document.metadata.get(
                "page", "Unknown"
            ),

            "sections": document.metadata.get(
                "sections", ""
            ),

            "relevance_score": round(
                float(score), 4
            )

        })

    return retrieved_documents


# =====================================================
# 7. RAG ANSWER GENERATION
# =====================================================

def ask_question_with_sources(query):

    if not query or not query.strip():

        return {
            "answer": "Please enter a valid question.",
            "sources": []
        }

    try:

        retrieved = retrieve_context(query)

        # No useful context found
        if not retrieved:

            return {

                "answer": (
                    "I could not find sufficiently relevant "
                    "information in the available legal documents "
                    "to answer this question reliably. "
                    "Please consult the relevant legal Act or a "
                    "qualified legal professional."
                ),

                "sources": []

            }

        # Build source-grounded context
        context_parts = []

        for index, item in enumerate(
            retrieved, start=1
        ):

            context_parts.append(

                f"""
SOURCE {index}

Document: {item['source']}
PDF Page: {item['page']}
Detected Sections: {item['sections']}

Legal Text:
{item['content']}
"""

            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are Women Law Assistant, an AI legal information
assistant specializing in Indian women's rights.

IMPORTANT RULES:

1. Answer using ONLY the retrieved legal context.
2. Do not invent statutory sections, punishments,
   penalties, procedures or legal provisions.
3. Never assume that a provision exists if it is not
   present in the supplied context.
4. Clearly distinguish between information explicitly
   mentioned in the legal text and any interpretation.
5. If the question cannot be answered from the context,
   explicitly state that the available documents do not
   contain sufficient information.
6. Explain legal concepts in simple, understandable language.
7. Mention the relevant Act and section only when supported
   by the retrieved text.
8. If multiple legal provisions are relevant, explain them
   separately.
9. Do not present yourself as a lawyer or claim to provide
   professional legal advice.
10. Do not treat retrieved text as proof that a law is
    currently in force or has not been amended.
11. If the context contains conflicting provisions, explain
    that the documents appear inconsistent rather than
    silently choosing one.

ANSWER FORMAT:

- Direct Answer
- Relevant Legal Provisions
- Explanation
- Practical Information (only if supported by context)
- Source Documents

Keep the answer clear, factual and appropriately detailed.
"""

        user_prompt = f"""
Retrieved Legal Context:

{context}

User Question:

{query}

Provide an accurate answer based strictly on the
retrieved legal context.
"""

        response = client.chat.completions.create(

            model=LLM_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.1

        )

        answer = response.choices[0].message.content

        sources = [

            {
                "source": item["source"],
                "page": item["page"],
                "sections": item["sections"],
                "relevance_score": item["relevance_score"]
            }

            for item in retrieved

        ]

        return {

            "answer": answer,

            "sources": sources

        }

    except Exception as e:

        print("RAG Error:", str(e))

        return {

            "answer": (
                "An error occurred while processing your "
                "question. Please try again."
            ),

            "sources": [],

            "error": str(e)

        }


# =====================================================
# 8. BACKWARD COMPATIBILITY
# =====================================================

def ask_question(query):
    """
    Preserves your old function interface.

    Existing code expecting a string answer
    can continue using ask_question().
    """

    result = ask_question_with_sources(query)

    return result["answer"]


# =====================================================
# 9. NORMAL LLM WITHOUT RAG
# =====================================================

def ask_normal_llm(query):
    """
    Baseline model for research comparison.

    No ChromaDB retrieval.
    No PDF context.
    """

    if not query or not query.strip():

        return "Please enter a valid question."

    response = client.chat.completions.create(

        model=LLM_MODEL,

        messages=[

            {
                "role": "system",

                "content": """
You are a general AI assistant.

Answer questions about Indian women's laws
using your pretrained knowledge.

Do not claim that you have searched or retrieved
the user's uploaded legal documents.

If you are uncertain, clearly state your uncertainty.
Distinguish general information from verified legal text.

This answer is generated WITHOUT retrieval-augmented
generation and must not be presented as verified
against the uploaded documents.
"""
            },

            {
                "role": "user",
                "content": query
            }

        ],

        temperature=0.1

    )

    return response.choices[0].message.content


# =====================================================
# 10. COMMAND LINE EXECUTION
# =====================================================

if __name__ == "__main__":

    print("\n WOMEN LAW RAG PIPELINE")

    print("1. Extract PDF text")
    print("2. Chunk extracted text")
    print("3. Generate embeddings")
    print("4. Ask RAG question")
    print("5. Ask normal LLM question")
    print("6. Exit")

    while True:

        choice = input(
            "\nSelect an option: "
        ).strip()

        if choice == "1":

            extract_text()

        elif choice == "2":

            chunk_text()

        elif choice == "3":

            create_embeddings()

        elif choice == "4":

            question = input(
                "\nEnter your legal question: "
            )

            result = ask_question_with_sources(
                question
            )

            print("\nRAG ANSWER:\n")
            print(result["answer"])

            print("\nRETRIEVED SOURCES:\n")

            for source in result["sources"]:
                print(source)

        elif choice == "5":

            question = input(
                "\nEnter your question: "
            )

            print("\nNORMAL LLM ANSWER:\n")

            print(ask_normal_llm(question))

        elif choice == "6":

            print("Goodbye!")
            break

        else:

            print("Invalid option. Try again.")
