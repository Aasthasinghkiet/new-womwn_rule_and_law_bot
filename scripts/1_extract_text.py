
import os
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm

# Input and output directories
pdf_dir = Path("data/pdfs/")
output_dir = Path("data/cleaned_texts/")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Find ALL PDFs, including PDFs inside subfolders
pdf_files = sorted(
    file for file in pdf_dir.rglob("*")
    if file.is_file() and file.suffix.lower() == ".pdf"
)

if not pdf_files:
    print("No PDF files found in:", pdf_dir)
    exit()

print(f"\nFound {len(pdf_files)} PDF documents.\n")

successful = 0
failed = 0
empty = 0

for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):

    try:
        reader = PdfReader(str(pdf_path))

        text = ""

        # Extract text from every page
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + " "

        # Clean extracted text
        clean_text = " ".join(text.split())

        # Keep output files in the existing flat folder
        # Use folder names to avoid duplicate filenames
        relative_path = pdf_path.relative_to(pdf_dir)

        if len(relative_path.parts) > 1:
            output_name = "__".join(
                relative_path.with_suffix("").parts
            ) + ".txt"
        else:
            output_name = pdf_path.stem + ".txt"

        output_file = output_dir / output_name

        # Save extracted text
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(clean_text)

        if clean_text:
            successful += 1
        else:
            empty += 1
            print(f"\nWarning: No extractable text in {pdf_path.name}")

    except Exception as e:
        failed += 1
        print(f"\nError processing {pdf_path.name}: {e}")

print("\n========== EXTRACTION SUMMARY ==========")
print(f"Total PDFs found: {len(pdf_files)}")
print(f"Successfully extracted: {successful}")
print(f"Empty PDFs: {empty}")
print(f"Failed PDFs: {failed}")
print("Output directory:", output_dir)
print("========================================")