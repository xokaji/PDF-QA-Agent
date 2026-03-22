import os
import re
from importlib import import_module
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older LangChain versions.
    RecursiveCharacterTextSplitter = import_module(
        "langchain.text_splitter"
    ).RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()


def load_pdf(path: str):
    print(f" Loading PDF: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"   Loaded {len(pages)} pages")
    return pages


def _find_pdf_files(directory: str):
    pdf_files = []
    for name in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, name)
        if os.path.isfile(full_path) and name.lower().endswith(".pdf"):
            pdf_files.append(full_path)
    return pdf_files


def resolve_pdf_path(user_input: str):
    path = user_input.strip().strip('"').strip("'")

    if not path:
        print(" Please enter a valid path.")
        return None

    if not os.path.exists(path):
        print(" File or folder not found!")
        return None

    if os.path.isfile(path):
        if not path.lower().endswith(".pdf"):
            print(" Please provide a .pdf file.")
            return None
        return path

    if not os.path.isdir(path):
        print(" Path must be a PDF file or a folder containing PDFs.")
        return None

    pdf_files = _find_pdf_files(path)
    if not pdf_files:
        print(" No PDF files found in that folder.")
        return None

    if len(pdf_files) == 1:
        selected = pdf_files[0]
        print(f" Found one PDF. Using: {selected}")
        return selected

    print(" Multiple PDF files found. Choose one:")
    for i, pdf in enumerate(pdf_files, start=1):
        print(f"   {i}. {os.path.basename(pdf)}")

    choice = input("Select file number: ").strip()
    if not choice.isdigit():
        print(" Invalid selection.")
        return None

    index = int(choice)
    if index < 1 or index > len(pdf_files):
        print(" Selection out of range.")
        return None

    return pdf_files[index - 1]


def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    print(f"   Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    print(" Building vector store...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your environment or .env file.")

    # Try likely available embedding models in order.
    candidate_models = [
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
        "embedding-001",
    ]
    last_error = None

    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            print(f"   Vector store ready! (embedding model: {model_name})")
            return vector_store
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to build embeddings with available models. "
        "Verify your Gemini API key and model access. "
        f"Last error: {last_error}"
    )


def _tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _keyword_retrieve(chunks, question: str, k: int = 4):
    q_tokens = _tokenize(question)
    scored = []
    for idx, doc in enumerate(chunks):
        d_tokens = _tokenize(getattr(doc, "page_content", ""))
        score = len(q_tokens.intersection(d_tokens))
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    top_docs = [doc for _, _, doc in scored[:k]]

    # If all scores are zero, still return the first k chunks as fallback context.
    if top_docs and scored[0][0] == 0:
        return chunks[:k]
    return top_docs


def _extractive_answer(question: str, docs):
    q_tokens = _tokenize(question)
    best_sentence = ""
    best_score = -1

    for doc in docs:
        text = getattr(doc, "page_content", "")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sent = sentence.strip()
            if not sent:
                continue
            score = len(q_tokens.intersection(_tokenize(sent)))
            if score > best_score:
                best_score = score
                best_sentence = sent

    if best_sentence:
        return (
            "Model response is unavailable, so this is an extractive answer from the document: "
            f"{best_sentence}"
        )

    return "Model response is unavailable and no matching text was found in the document chunks."


def _build_chat_model(api_key: str):
    candidate_models = [
        "gemini-2.0-flash",
        "models/gemini-2.0-flash",
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-pro",
    ]
    last_error = None

    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
            )
            # Probe once during startup so bad model IDs fail early.
            llm.invoke("Reply with: OK")
            print(f"   Chat model ready! ({model_name})")
            return llm
        except Exception as exc:
            last_error = exc

    print(
        " Warning: no supported Gemini chat model available. "
        "Falling back to extractive answers. "
        f"Last error: {last_error}"
    )
    return None


def build_qa_chain(vector_store):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = _build_chat_model(api_key) if api_key else None
    if not api_key:
        print(" Warning: GEMINI_API_KEY missing. Falling back to extractive answers.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) if vector_store else None
    return llm, retriever


def ask_question(llm, retriever, question: str, chunks):
    if retriever is not None:
        docs = retriever.invoke(question)
    else:
        docs = _keyword_retrieve(chunks, question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "Answer the user's question using only the provided context. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    if llm is None:
        answer = _extractive_answer(question, docs)
        return answer, docs

    try:
        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))
    except Exception as exc:
        answer = (
            "I could not call the chat model right now. "
            f"Details: {exc}\n"
            + _extractive_answer(question, docs)
        )
    return answer, docs


def main():
    user_input = input("Enter the path to your PDF file (or folder with PDFs): ")
    pdf_path = resolve_pdf_path(user_input)
    if not pdf_path:
        return

    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)
    store = None
    try:
        store = create_vector_store(chunks)
    except Exception as exc:
        print(f" Warning: embedding search unavailable, using keyword retrieval. Details: {exc}")

    llm, retriever = build_qa_chain(store)

    print("\n Agent ready! Type your questions (or 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer, source_docs = ask_question(llm, retriever, question, chunks)
        print(f"\nAgent: {answer}")

        sources = {doc.metadata.get("page", "?") for doc in source_docs}
        print(f"  (Sources: pages {sorted(sources)})\n")

if __name__ == "__main__":
    main()