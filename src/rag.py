"""
RAG module for the EV Business Research Agent.

Pipeline:
    PDF documents
        ↓
    Text extraction
        ↓
    Document chunking
        ↓
    Sentence-transformer embeddings
        ↓
    Chroma vector database
        ↓
    Similarity retrieval

The module preserves source, document, category,
and page-level metadata for research provenance.

The vector database is built automatically when it
does not already exist.
"""

from pathlib import Path
from typing import List, Dict
import shutil

import torch
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# Keep the capitalization exactly as used in the repository.
DOCUMENTS_DIR = PROJECT_ROOT / "Documents"

VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ev_business_research"


# ============================================================
# RAG CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 5


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_pdf_documents() -> List[Document]:
    """
    Extract text from every PDF in the document corpus.

    Each page becomes one LangChain Document so that
    page-level provenance can be preserved.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Research documents directory not found:\n"
            f"{DOCUMENTS_DIR}\n\n"
            f"Make sure the repository contains the Documents/ folder."
        )

    pdf_files = sorted(
        DOCUMENTS_DIR.rglob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside:\n"
            f"{DOCUMENTS_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:

        category = pdf_path.parent.name

        try:

            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text()

                if text and text.strip():

                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "category": category,
                                "file_name": pdf_path.name,
                                "page": page_number,
                                "source": pdf_path.name,
                            },
                        )
                    )

        except Exception as exc:

            print(
                f"Warning: failed to process "
                f"{pdf_path.name}: {exc}"
            )

    if not documents:
        raise ValueError(
            "No readable PDF text was extracted "
            "from the research corpus."
        )

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def chunk_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split extracted documents into overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    if not chunks:
        raise ValueError(
            "Document chunking produced no chunks."
        )

    return chunks


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Load the Sentence Transformer embedding model.

    Uses GPU when available and CPU otherwise.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    return embedding_model


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vector_database(
    chunks: List[Document],
    embedding_model=None,
    batch_size: int = 256,
):
    """
    Create a Chroma vector database from document chunks.

    The database is persisted locally under the project
    directory so it can be reused by the application.
    """

    if not chunks:
        raise ValueError(
            "Cannot build vector database "
            "from an empty chunk list."
        )

    if embedding_model is None:
        embedding_model = create_embedding_model()

    # Always start with a clean database when rebuilding.
    # This prevents duplicate chunks if the build function
    # is accidentally executed more than once.
    if VECTOR_DB_DIR.exists():
        shutil.rmtree(
            VECTOR_DB_DIR,
            ignore_errors=True,
        )

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            VECTOR_DB_DIR
        )
    )

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(chunks),
        )

        print(
            f"Embedding chunks "
            f"{start + 1:,}-{end:,} "
            f"of {len(chunks):,}..."
        )

        vector_db.add_documents(
            chunks[start:end]
        )

    return vector_db


# ============================================================
# AUTOMATIC VECTOR DATABASE INITIALIZATION
# ============================================================

def _build_vector_database_from_documents(
    embedding_model=None,
):
    """
    Build a fresh Chroma database from the Documents/ corpus.
    """

    print(
        "Building Chroma database "
        "from research documents..."
    )

    documents = load_pdf_documents()

    print(
        f"Extracted {len(documents):,} "
        f"page-level documents."
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"Created {len(chunks):,} "
        f"document chunks."
    )

    vector_db = build_vector_database(
        chunks,
        embedding_model,
    )

    print(
        "Chroma vector database created successfully."
    )

    return vector_db


# ============================================================
# LOAD OR BUILD VECTOR DATABASE
# ============================================================

def load_vector_database(
    embedding_model=None,
):
    """
    Load the existing Chroma vector database.

    If the database does not exist, automatically build it
    from the PDFs inside Documents/.

    This makes the project reproducible from a clean clone:
        git clone
        install dependencies
        add API key
        streamlit run app.py
    """

    if embedding_model is None:
        embedding_model = create_embedding_model()

    # --------------------------------------------------------
    # Case 1: Chroma database does not exist
    # --------------------------------------------------------

    if not VECTOR_DB_DIR.exists():

        print(
            "Chroma database not found. "
            "Building it automatically..."
        )

        return _build_vector_database_from_documents(
            embedding_model
        )

    # --------------------------------------------------------
    # Case 2: Chroma database exists
    # --------------------------------------------------------

    try:

        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(
                VECTOR_DB_DIR
            ),
        )

        count = vector_db._collection.count()

        if count > 0:

            print(
                f"Existing Chroma database loaded "
                f"({count:,} chunks)."
            )

            return vector_db

        # Database directory exists but collection is empty.
        print(
            "Chroma database exists but is empty. "
            "Rebuilding it..."
        )

        shutil.rmtree(
            VECTOR_DB_DIR,
            ignore_errors=True,
        )

        return _build_vector_database_from_documents(
            embedding_model
        )

    except Exception as exc:

        print(
            f"Existing Chroma database could not be loaded: "
            f"{exc}"
        )

        print(
            "Rebuilding Chroma database from Documents/..."
        )

        shutil.rmtree(
            VECTOR_DB_DIR,
            ignore_errors=True,
        )

        return _build_vector_database_from_documents(
            embedding_model
        )


# ============================================================
# RESEARCH / RAG TOOL
# ============================================================

def research_rag_tool(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
    vector_db=None,
) -> Dict:
    """
    Retrieve relevant research evidence.

    Returns:
        query
        result_count
        results containing:
            - content
            - source
            - page
            - category
    """

    if not query or not query.strip():
        raise ValueError(
            "Research query cannot be empty."
        )

    k = max(
        1,
        min(int(k), 8),
    )

    if vector_db is None:
        vector_db = load_vector_database()

    retrieved_docs = vector_db.similarity_search(
        query,
        k=k,
    )

    if not retrieved_docs:

        return {
            "status": "success",
            "query": query,
            "results": [],
            "result_count": 0,
            "message": "No relevant information found.",
        }

    results = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1,
    ):

        metadata = doc.metadata

        results.append(
            {
                "result_number": index,
                "content": doc.page_content,
                "source": metadata.get(
                    "source",
                    metadata.get(
                        "file_name",
                        "Unknown",
                    ),
                ),
                "page": metadata.get(
                    "page",
                    "Unknown",
                ),
                "category": metadata.get(
                    "category",
                    "Unknown",
                ),
            }
        )

    return {
        "status": "success",
        "query": query,
        "results": results,
        "result_count": len(results),
    }


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

def initialize_rag():
    """
    Initialize the complete RAG pipeline.

    If a Chroma database already exists, load it.

    Otherwise:
        1. Read PDFs from Documents/
        2. Extract page-level text
        3. Split text into chunks
        4. Generate embeddings
        5. Build and persist Chroma

    Returns:
        vector_db
    """

    return load_vector_database()