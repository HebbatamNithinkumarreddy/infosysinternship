# ---------------- IMPORTS ----------------
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from pathlib import Path
import os
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()   # adjust path if needed
print("Loaded GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))  # Debug
print("Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
# ---------------- CONFIG ----------------
DOCS_PATH = Path("./my_docs")         # folder or single file (.txt, .md, .pdf)
INDEX_PATH = Path("./faiss_index")    # where FAISS index is stored
REBUILD_INDEX = True                  # True = rebuild from docs; False = load
EMBED_MODEL = "models/embedding-001"  # Gemini embedding model
CHAT_MODEL = "llama-3.3-70b-versatile" # Groq chat model
TOP_K = 4                             # how many chunks to retrieve
SEARCH_TYPE = "mmr"                   # "mmr" | "similarity"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

QUESTION = " how the Gen-AI can help with education in many ways."
# -----------------------------------------

# STEP 1) ENSURE API KEYS
if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit(" GOOGLE_API_KEY is not set. Check your .env file and path.")

if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit(" GROQ_API_KEY is not set. Check your .env file and path.")

# STEP 2) FIND & LOAD DOCUMENTS
def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = [".txt", ".md", ".pdf"]
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in (".txt", ".md"):
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

# STEP 3) SPLIT DOCS INTO CHUNKS
def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

# STEP 4) MAKE / LOAD FAISS VECTOR STORE
def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    if rebuild:
        print("Building FAISS index from documents...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        print(f"Saved index to: {INDEX_PATH.resolve()}")
    print(f"Loading FAISS index from: {INDEX_PATH.resolve()}")
    vs = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index.")
    return vs

# STEP 5) BUILD THE RETRIEVER
def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K},
    )

# STEP 6) CREATE THE RAG CHAIN
def make_rag_chain(retriever):
    # a) Define prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a concise, careful assistant. Answer ONLY from the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources by filename and, if present, page."
        ),
        ("human", "Question:\n{input}\n\nContext:\n{context}"),
    ])

    # b) Chat model (Groq - LLaMA 3.3)
    llm = ChatGroq(model=CHAT_MODEL, temperature=0.2)

    # c) Combine docs into prompt
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # d) RAG chain
    return create_retrieval_chain(retriever, doc_chain)

# STEP 7) PRETTY-PRINT SOURCES
def format_sources(ctx: list[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        name = Path(src).name
        lines.append(f"{name}" + (f" (page {page})" if page is not None else ""))
    return "\n".join(lines)

# STEP 8) GLUE IT ALL TOGETHER
def main():
    # 8a) Rebuild / load documents
    chunks: list[Document] = []
    if REBUILD_INDEX:
        print(f"Scanning docs under: {DOCS_PATH.resolve()}")
        files = find_files(DOCS_PATH)
        if not files:
            raise SystemExit("No .txt/.md/.pdf files found.")
        print(f"Loading {len(files)} files...")
        docs = load_documents(files)

        print(f"Splitting {len(docs)} docs (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        chunks = split_documents(docs)

    # 8b) Build / load FAISS
    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)

    # 8c) Make retriever + RAG
    retriever = make_retriever(vectorstore)
    rag = make_rag_chain(retriever)

    # 8d) Ask one question
    if QUESTION:
        print(f"Question: {QUESTION}")
        result = rag.invoke({"input": QUESTION})
        answer = result.get("answer") or result.get("output") or str(result)
        print("\nAnswer:\n" + answer.strip())

        ctx = result.get("context", [])
        if ctx:
            print("\nSources:")
            print(format_sources(ctx))

if __name__ == "__main__":
    main()