# Women Law and Rules Chatbot ⚖️

An AI-powered Women Law and Rules Chatbot built using Retrieval-Augmented Generation (RAG). It helps users understand women's legal rights, laws, and regulations by providing answers based on relevant legal documents.

## 🚀 Features

- AI-powered legal question answering.
- Retrieval-Augmented Generation (RAG).
- ChromaDB vector database for storing document embeddings.
- Semantic search for retrieving relevant legal information.
- FastAPI backend.
- Interactive chatbot interface.
- Supports women's rights and legal awareness.

## 🛠️ Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- Hugging Face / Embedding Models
- LLM API (as configured)
- HTML, CSS, JavaScript

## 📁 Project Structure

```text
women-law-and-rules-bot/
│
├── scripts/
│   └── 3_generate_embeddings.py
│
├── embeddings/
│   └── vectors/             # Generated ChromaDB (not included)
│
├── data/                    # Legal documents
│
├── main.py                  # FastAPI backend
├── rag_pipeline.py          # RAG pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aasthasinghkiet/new-womwn_rule_and_law_bot.git
```

```bash
cd new-womwn_rule_and_law_bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the API credentials required by your configured LLM provider.

Do not upload your API keys to GitHub.

### 5. Generate ChromaDB Vector Database

The ChromaDB database is not included in this repository because of GitHub file-size restrictions.

After cloning the repository, generate the database locally:

```bash
python scripts/3_generate_embeddings.py
```

This script processes the available legal documents, generates embeddings, and stores them in:

```text
embeddings/vectors/
```

Make sure the original legal documents are available in the required data folder before running this script.

### 6. Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Launch the Frontend

Open the frontend application in your browser or start its development server according to the frontend setup.

## 🔄 How the RAG System Works

1. User enters a legal question.
2. The question is converted into embeddings.
3. ChromaDB retrieves relevant legal document chunks.
4. The retrieved context is passed to the LLM.
5. The LLM generates a context-based response.
6. The chatbot displays the answer to the user.

## ⚠️ Important Notes

- Generate ChromaDB embeddings before running the chatbot.
- The vector database is excluded from GitHub.
- Keep your API keys private.
- Ensure all required legal documents are available before generating embeddings.
- This chatbot is intended for legal awareness and informational purposes, not as a substitute for professional legal advice.

## 👩‍💻 Author

**Aastha Singh**

GitHub: https://github.com/Aasthasinghkiet
