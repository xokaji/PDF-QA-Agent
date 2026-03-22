# PDF QA Agent

A simple command-line app that answers questions from a PDF file.

## Features

- Load a PDF file (or a folder containing PDFs)
- Split content into chunks for retrieval
- Use Gemini embeddings + FAISS when available
- Fall back to keyword retrieval if embeddings are unavailable
- Show source page numbers with each answer

## Requirements

- Python 3.10+
- A Gemini API key

Install dependencies:

```bash
pip install python-dotenv langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf
```

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run

```bash
python pdf_agent.py
```

Then enter either:

- A direct path to a `.pdf` file, or
- A folder path containing `.pdf` files

Type `quit` to exit.

## Notes

- If chat model access is unavailable, the app falls back to extractive answers from document text.
- If embedding models are unavailable, the app falls back to keyword-based retrieval.
