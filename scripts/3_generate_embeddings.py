
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Please check your .env file."
    )

# Folder paths - existing paths preserved
chunks_dir = Path("data/chunks/")
persist_dir = "embeddings/vectors/"

os.makedirs(persist_dir, exist_ok=True)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)

# Collection name shared with the RAG pipeline
collection_name = "women_law_collection"

documents = []

# Read all chunk JSON files
json_files = sorted(chunks_dir.glob("*.json"))

if not json_files:
    raise ValueError(
        "No chunk JSON files found. Run 2_chunk_text.py first."
    )

print(f"\nFound {len(json_files)} chunk files.")

for file_path in tqdm(json_files, desc="Reading chunks"):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:

            content = item.get("content", "").strip()

            if not content:
                continue

            documents.append({
                "content": content,
                "metadata": {
                    "source": item.get("source", file_path.name),
                    "category": item.get("category", "General"),
                    "chunk_index": int(item.get("chunk_index", 0))
                }
            })

    except Exception as e:
        print(f"\nError reading {file_path.name}: {e}")

if not documents:
    raise ValueError("No valid text chunks available for embedding.")

texts = [doc["content"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

print(f"\nTotal chunks to embed: {len(texts)}")

# Rebuild only our named collection.
# Other collections in the existing Chroma database are untouched.
try:
    old_db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    old_db.delete_collection()
    print("Previous Women Law collection cleared.")

except Exception:
    print("No previous Women Law collection found.")

# Create a fresh vector database
db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    collection_name=collection_name,
    persist_directory=persist_dir
)

# Chroma automatically persists data to disk
print("\n========== EMBEDDING SUMMARY ==========")
print("Total documents:", len(json_files))
print("Total embedded chunks:", len(texts))
print("Embedding model: text-embedding-3-large")
print("Collection:", collection_name)
print("Vector database:", persist_dir)
print("======================================")
print("\nEmbeddings generated successfully!")
