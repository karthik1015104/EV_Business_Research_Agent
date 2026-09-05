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
"""

from pathlib import Path
from typing import List, Dict

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

DOCUMENTS_DIR = (
    PROJECT_ROOT / "documents"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT / "chroma_db"
)

COLLECTION_NAME = (
    "ev_business_research"
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

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
            f"Documents directory not found:\n"
            f"{DOCUMENTS_DIR}"
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

            reader = PdfReader(
                str(pdf_path)
            )

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
                                "source": pdf_path.name
                            }
                        )
                    )

        except Exception as exc:

            print(
                f"Warning: failed to process "
                f"{pdf_path.name}: {exc}"
            )

    if not documents:
        raise ValueError(
            "No readable PDF text was extracted."
        )

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def chunk_documents(
    documents: List[Document]
) -> List[Document]:
    """
    Split extracted documents into overlapping chunks.
    """

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
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

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )
    )

    return embedding_model


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vector_database(
    chunks: List[Document],
    embedding_model=None,
    batch_size: int = 256
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
        embedding_model = (
            create_embedding_model()
        )

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
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
        batch_size
    ):

        end = min(
            start + batch_size,
            len(chunks)
        )

        vector_db.add_documents(
            chunks[start:end]
        )

    return vector_db


# ============================================================
# LOAD EXISTING VECTOR DATABASE
# ============================================================

def load_vector_database(
    embedding_model=None
):
    """
    Load an existing Chroma vector database.

    Raises an error if the database has not
    been built yet.
    """

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not found.\n"
            "Run build_vector_database() first."
        )

    if embedding_model is None:
        embedding_model = (
            create_embedding_model()
        )

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            VECTOR_DB_DIR
        )
    )

    if vector_db._collection.count() == 0:
        raise ValueError(
            "Chroma database exists but "
            "contains no documents."
        )

    return vector_db


# ============================================================
# RESEARCH / RAG TOOL
# ============================================================

def research_rag_tool(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
    vector_db=None
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
        min(int(k), 8)
    )

    if vector_db is None:
        vector_db = load_vector_database()

    retrieved_docs = (
        vector_db.similarity_search(
            query,
            k=k
        )
    )

    if not retrieved_docs:

        return {
            "status": "success",
            "query": query,
            "results": [],
            "result_count": 0,
            "message": (
                "No relevant information found."
            )
        }

    results = []

    for index, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        metadata = doc.metadata

        results.append({
            "result_number": index,
            "content": doc.page_content,
            "source": metadata.get(
                "source",
                metadata.get(
                    "file_name",
                    "Unknown"
                )
            ),
            "page": metadata.get(
                "page",
                "Unknown"
            ),
            "category": metadata.get(
                "category",
                "Unknown"
            )
        })

    return {
        "status": "success",
        "query": query,
        "results": results,
        "result_count": len(results)
    }


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

def initialize_rag():
    """
    Initialize the complete RAG pipeline.

    If a Chroma database already exists, it is loaded.
    Otherwise, PDFs are extracted, chunked, embedded,
    and stored in a new Chroma database.

    Returns:
        vector_db
    """

    embedding_model = (
        create_embedding_model()
    )

    if VECTOR_DB_DIR.exists():

        try:

            vector_db = load_vector_database(
                embedding_model
            )

            print(
                "✓ Existing Chroma database loaded."
            )

            return vector_db

        except (
            FileNotFoundError,
            ValueError
        ):

            pass

    print(
        "Building Chroma database "
        "from research documents..."
    )

    documents = load_pdf_documents()

    print(
        f"✓ Extracted {len(documents):,} "
        "page-level documents."
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"✓ Created {len(chunks):,} "
        "document chunks."
    )

    vector_db = build_vector_database(
        chunks,
        embedding_model
    )

    print(
        "✓ Chroma vector database created."
    )

    return vector_db