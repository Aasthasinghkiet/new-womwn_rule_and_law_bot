
import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Existing folder paths - kept unchanged
input_dir = Path("data/cleaned_texts/")
output_dir = Path("data/chunks/")

output_dir.mkdir(parents=True, exist_ok=True)

# Improved legal text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "; ",
        " ",
        ""
    ],
    length_function=len
)

# Read all extracted text files
text_files = sorted(input_dir.glob("*.txt"))

if not text_files:
    print("No text files found in:", input_dir)
    exit()

print(f"\nFound {len(text_files)} text documents.")

total_chunks = 0
failed_files = 0

for file_path in tqdm(text_files, desc="Creating chunks"):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"\nSkipping empty file: {file_path.name}")
            continue

        chunks = splitter.split_text(text)

        # Extract category from folder-prefixed filenames
        # Example: Dowry__Dowry_Act.txt -> Dowry
        if "__" in file_path.stem:
            category = file_path.stem.split("__")[0]
        else:
            category = "General"

        data = []

        for index, chunk in enumerate(chunks):

            data.append({
                "source": file_path.name,
                "category": category,
                "chunk_index": index,
                "content": chunk
            })

        # Preserve the existing JSON output structure
        output_file = output_dir / f"{file_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        total_chunks += len(chunks)

    except Exception as e:
        failed_files += 1
        print(f"\nError processing {file_path.name}: {e}")

print("\n========== CHUNKING SUMMARY ==========")
print("Documents found:", len(text_files))
print("Total chunks created:", total_chunks)
print("Failed documents:", failed_files)
print("Output directory:", output_dir)
print("======================================")
