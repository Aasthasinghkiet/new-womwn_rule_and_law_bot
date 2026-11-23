
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-large"
)

PERSIST_DIR = os.getenv(
    "CHROMA_PATH",
    "embeddings/vectors/"
)

COLLECTION_NAME = "women_law_collection"

# ==========================================
# 2. INITIALIZE OPENAI AND CHROMA
# ==========================================

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=api_key
)

db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

client = OpenAI(api_key=api_key)

# ==========================================
# 3. LEGAL ASSISTANT INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are Women Law & Rights Assistant, a legal information
assistant specializing in Indian laws concerning women.

Your primary responsibility is to provide accurate,
clear, meaningful and source-supported information.

STRICT RULES:

1. Answer using ONLY the retrieved document context.

2. Never invent:
   - Legal sections
   - Punishments
   - Fines
   - Court procedures
   - Complaint procedures
   - Legal rights
   - Government schemes

3. Preserve the exact legal meaning of the source.
   Pay special attention to:
   - Minimum and maximum punishment
   - Mandatory and optional provisions
   - Exceptions and conditions
   - Time limits
   - Eligibility requirements

4. Give the name of the relevant Act and section
   only when supported by the retrieved context.

5. Cite supporting documents using references such as
   [S1], [S2], etc.

6. Distinguish between:
   - Acts and statutory provisions
   - Rules
   - Government handbooks
   - Explanatory guides

   Do not present guidance from a handbook as statutory
   wording.

7. If multiple documents are relevant, combine their
   information carefully without contradicting them.

8. If the retrieved context does not contain sufficient
   information, clearly say:

   "I don't have sufficient information in the available
   legal documents to answer this question accurately."

9. Do not use your general knowledge to fill missing
   legal information.

10. Explain legal concepts in simple language so that
    ordinary users can understand them.

11. For practical questions, organize the answer into:
    - Applicable law
    - Relevant rights or protections
    - Available remedy or procedure, if documented
    - Important conditions

12. Do not claim to be a lawyer or provide a guarantee
    about the outcome of a legal case.

13. If the question concerns immediate danger or violence,
    acknowledge the urgency and recommend contacting
    appropriate emergency services. Do not invent
    document-based emergency procedures.

14. Answer in the language used by the user whenever
    possible.

15. Do not assume that every retrieved document is
    current law. Mention when a source's date or current
    legal status cannot be established from the context.
"""

# ==========================================
# 4. RETRIEVE RELEVANT LEGAL DOCUMENTS
# ==========================================

def retrieve_context(query, k=6):

    results = db.similarity_search(
        query,
        k=k
    )

    if not results:
        return "", []

    context_parts = []
    sources = []
    source_map = {}

    for doc in results:

        metadata = doc.metadata or {}

        source_name = metadata.get(
            "source",
            "Unknown document"
        )

        category = metadata.get(
            "category",
            "General"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        # Give each unique document a source reference
        if source_name not in source_map:

            source_id = f"S{len(source_map) + 1}"
            source_map[source_name] = source_id

            sources.append({
                "id": source_id,
                "name": source_name,
                "category": category
            })

        source_id = source_map[source_name]

        context_parts.append(
            f"""
            [Source: {source_id}]
            Document: {source_name}
            Category: {category}
            Chunk: {chunk_index}

            Content:
            {doc.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    return context, sources


# ==========================================
# 5. GENERATE LEGAL ANSWER
# ==========================================

def ask_question(query):

    query = query.strip()

    if not query:
        return "Please enter a valid question."

    try:

        # Retrieve relevant chunks
        context, sources = retrieve_context(
            query,
            k=6
        )

        if not context:
            return (
                "I couldn't find relevant information "
                "in the available legal documents."
            )

        # Prepare user prompt
        user_prompt = f"""
        Retrieved legal document context:

        {context}

        User question:
        {query}

        Instructions:
        Answer the question strictly using the retrieved
        context and follow all legal assistant rules.

        Include supporting source references such as [S1].
        If the information is insufficient, clearly state
        that rather than guessing.
        """

        # Generate response
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0,
            max_tokens=1200
        )

        answer = response.choices[0].message.content

        if not answer:
            return "Unable to generate an answer. Please try again."

        # Add retrieved source information
        source_text = "\n\nSources used:\n"

        for source in sources:
            source_text += (
                f"\n[{source['id']}] "
                f"{source['name']} "
                f"({source['category']})"
            )

        final_answer = answer.strip() + source_text

        return final_answer

    except Exception as e:

        print("RAG ERROR:", str(e))

        return (
            "Sorry, an error occurred while processing "
            "your question. Please try again."
        )


# ==========================================
# 6. TERMINAL CHAT INTERFACE
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print(" WOMEN LAW & RIGHTS CHATBOT")
    print(" Indian Women's Legal Information")
    print("====================================")

    print("\nType 'exit' to stop the chatbot.")

    while True:

        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("\nThank you for using Women Law Bot.")
            break

        if not query:
            print("Please enter a question.")
            continue

        answer = ask_question(query)

        print("\nAssistant:\n")
        print(answer)

        print("\n------------------------------------")
