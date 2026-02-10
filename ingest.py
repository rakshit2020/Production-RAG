"""
Data Ingestion Pipeline for RAG with NVIDIA NIM + Milvus GPU
Loads markdown files, splits them intelligently, and stores in Milvus.
"""

import os
import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_milvus import Milvus

# ============== CONFIGURATION ==============
class Config:
    # Milvus Settings
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "rag_documents"
    
    # NVIDIA NIM Settings (API Catalog or Self-Hosted)
    NVIDIA_API_KEY = "your-nvidia-api-key"  # Set your key here or via env
    EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    
    # For Self-Hosted NIM (uncomment and modify if using local NIM)
    # EMBEDDING_BASE_URL = "http://localhost:8080/v1"
    
    # Document Processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    MARKDOWN_FOLDER = r"D:\DATA\TalkwithDATA"  # Your folder path

    # HNSW Index Parameters (Higher values = Better accuracy, More memory/time)
    INDEX_TYPE = "HNSW"
    INDEX_PARAMS = {
        "M": 32,              # Max connections per node (16-64, higher = more accurate)
        "efConstruction": 256  # Build-time search depth (64-512, higher = better quality index)
    }
    
    # Alternative: FLAT index for highest accuracy (brute force - slower)
    # INDEX_TYPE = "FLAT"
    # INDEX_PARAMS = {}

# ============== LOGGING ==============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== DOCUMENT PROCESSING ==============
def split_markdown_tables(text: str) -> List[str]:
    """Splits markdown into table blocks and normal text blocks."""
    table_pattern = r"(\n\|.*?\|\n(?:\|.*?\|\n)+)"
    parts = re.split(table_pattern, text, flags=re.DOTALL)
    
    blocks = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks


def load_and_process_markdown(file_path: str) -> List[Document]:
    """Load a markdown file and process it into chunks."""
    logger.info(f"Loading file: {file_path}")
    
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    raw_text = documents[0].page_content
    
    # Split into blocks (tables vs regular text)
    blocks = split_markdown_tables(raw_text)
    docs = [Document(page_content=block, metadata={"source": file_path}) for block in blocks]
    
    # Text splitter for non-table content
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language="markdown",
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    final_docs = []
    for doc in docs:
        if doc.page_content.strip().startswith("|"):
            # Keep tables as-is
            final_docs.append(doc)
        else:
            # Split regular text
            final_docs.extend(text_splitter.split_documents([doc]))
    
    logger.info(f"Created {len(final_docs)} chunks from {file_path}")
    return final_docs


def load_all_markdown_files(folder_path: str) -> List[Document]:
    """Load all markdown files from a folder."""
    all_docs = []
    folder = Path(folder_path)
    
    md_files = list(folder.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {folder_path}")
    
    for md_file in md_files:
        try:
            docs = load_and_process_markdown(str(md_file))
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error processing {md_file}: {e}")
    
    return all_docs


def get_embeddings():
    """Initialize NVIDIA embeddings (API Catalog or Self-Hosted)."""
    # Option 1: NVIDIA API Catalog
    embeddings = NVIDIAEmbeddings(
        model=Config.EMBEDDING_MODEL,
        truncate="END"
    )
    
    # Option 2: Self-Hosted NIM (uncomment to use)
    # embeddings = NVIDIAEmbeddings(
    #     base_url=Config.EMBEDDING_BASE_URL,
    #     truncate="END"
    # )
    
    return embeddings


def ingest_to_milvus(documents: List[Document]) -> Milvus:
    """Ingest documents into Milvus vector store."""
    logger.info(f"Ingesting {len(documents)} documents to Milvus...")
    
    embeddings = get_embeddings()
    
    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection_args={"uri": Config.MILVUS_URI},
        drop_old=True, 
        index_params={
            "index_type": Config.INDEX_TYPE,
            "metric_type": "COSINE",  # or "L2", "IP" (Inner Product)
            "params": Config.INDEX_PARAMS
        }
    ) # Set False to append to existing collection
    
    logger.info(f"Successfully ingested documents to collection: {Config.COLLECTION_NAME}")
    return vectorstore


# ============== MAIN ==============
def main():
    # Set API Key
    os.environ["NVIDIA_API_KEY"] = Config.NVIDIA_API_KEY
    
    # Load all markdown files
    all_documents = load_all_markdown_files(Config.MARKDOWN_FOLDER)
    logger.info(f"Total documents to ingest: {len(all_documents)}")
    
    if not all_documents:
        logger.warning("No documents found. Check your folder path.")
        return
    
    # Ingest to Milvus
    ingest_to_milvus(all_documents)
    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()