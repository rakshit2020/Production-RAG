"""
Data Ingestion Pipeline V2 for Normalized Chandra RAG

Uses our improved chunking strategy + NVIDIA NIM + Milvus

This is a NEW file - does not modify existing ingest.py
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our improved chunker
from improved_chunking import ImprovedChunker

# LangChain imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus

import warnings
warnings.filterwarnings('ignore', message='.*AsyncMilvusClient.*')

# ============== CONFIGURATION ==============
class Config:
    # Milvus Settings
    MILVUS_URI = "http://localhost:19530"  # Your running Milvus
    COLLECTION_NAME = "normalized_chandra_v2"

    # NVIDIA NIM Settings
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "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")
    EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"

    # Document Processing - use our improved chunking
    CHUNK_MAX_SIZE = 2000

    # Folder with all files
    MARKDOWN_FOLDER = "/home/rakshit/Desktop/COOKING/PROJECTS/LangChain_Setup/RAG_scripts_CAIR/Normalize_ChandraOCR"

    # Index Parameters
    INDEX_TYPE = "HNSW"
    METRIC_TYPE = "COSINE"
    INDEX_PARAMS = {
        "M": 16,
        "efConstruction": 128
    }

    # Milvus VARCHAR limit (65KB max)
    MAX_TEXT_LENGTH = 60000  # Keep under limit


# ============== LOGGING ==============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== CONVERTER ==============
def truncate_content(content: str, max_length: int = 60000) -> str:
    """Truncate content if it exceeds Milvus VARCHAR limit"""
    if len(content) > max_length:
        return content[:max_length] + "\n\n[Content truncated due to length...]"
    return content


def clean_metadata(chunk) -> dict:
    """Clean metadata to ensure valid types for Milvus"""
    year = chunk.metadata.get("year")
    month = chunk.metadata.get("month")

    if year is None:
        year = -1
    if month is None:
        month = ""
    else:
        month = str(month)

    return {
        "chunk_id": str(chunk.chunk_id),
        "chunk_type": str(chunk.chunk_type),
        "parent_id": str(chunk.parent_id) if chunk.parent_id else "",
        "year": int(year),
        "month": month,
        "report_type": str(chunk.metadata.get("report_type", "unknown")),
    }


def chunk_to_langchain_doc(chunks: List, parent_chunks: List) -> List[Document]:
    """Convert our chunks to LangChain Document format"""
    docs = []

    # Add child chunks
    for chunk in chunks:
        meta = clean_metadata(chunk)
        meta["is_parent"] = False

        # Truncate content (Milvus VARCHAR limit is 65535)
        content = truncate_content(chunk.content, Config.MAX_TEXT_LENGTH)

        doc = Document(page_content=content, metadata=meta)
        docs.append(doc)

    # Add parent chunks
    for chunk in parent_chunks:
        meta = clean_metadata(chunk)
        meta["is_parent"] = True

        # Truncate content (Milvus VARCHAR limit is 65535)
        content = truncate_content(chunk.content, Config.MAX_TEXT_LENGTH)

        doc = Document(page_content=content, metadata=meta)
        docs.append(doc)

    return docs


def get_embeddings():
    """Initialize NVIDIA embeddings"""
    os.environ["NVIDIA_API_KEY"] = Config.NVIDIA_API_KEY

    embeddings = NVIDIAEmbeddings(
        model=Config.EMBEDDING_MODEL,
        truncate="END"
    )
    return embeddings


# ============== MAIN INGESTION ==============
def ingest_documents(file_paths: List[str], drop_old: bool = True):
    """Main ingestion pipeline"""

    logger.info(f"Starting ingestion for {len(file_paths)} files")
    print(f"\n{'='*60}")
    print("DATA INGESTION PIPELINE V2")
    print(f"{'='*60}")

    # Step 1: Chunk using improved chunker
    print("\n📝 Step 1: Chunking documents with improved strategy...")
    chunker = ImprovedChunker(max_chunk_size=Config.CHUNK_MAX_SIZE)
    results = chunker.process_documents(file_paths)

    child_chunks = results['child_chunks']
    parent_chunks = results['parent_chunks']

    print(f"   - Documents: {len(results['documents'])}")
    print(f"   - Child chunks: {len(child_chunks)}")
    print(f"   - Parent chunks: {len(parent_chunks)}")

    # Step 2: Convert to LangChain format
    print("\n🔄 Step 2: Converting to LangChain format...")
    docs = chunk_to_langchain_doc(child_chunks, parent_chunks)
    print(f"   - Total LangChain docs: {len(docs)}")

    # Check for large docs
    large_docs = [d for d in docs if len(d.page_content) > 60000]
    print(f"   - Large docs (>60K chars): {len(large_docs)}")

    # Step 3: Initialize embeddings
    print("\n🔗 Step 3: Initializing NVIDIA NIM embeddings...")
    embeddings = get_embeddings()
    print(f"   - Model: {Config.EMBEDDING_MODEL}")

    # Step 4: Ingest to Milvus
    print("\n🗄️  Step 4: Ingesting to Milvus...")
    print(f"   - Collection: {Config.COLLECTION_NAME}")
    print(f"   - Milvus URI: {Config.MILVUS_URI}")

    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection_args={"uri": Config.MILVUS_URI},
        drop_old=drop_old,
        index_params={
            "index_type": Config.INDEX_TYPE,
            "metric_type": Config.METRIC_TYPE,
            "params": Config.INDEX_PARAMS
        }
    )

    print(f"\n✅ INGESTION COMPLETE!")
    print(f"   - Total documents: {len(docs)}")
    print(f"   - Collection: {Config.COLLECTION_NAME}")

    return vectorstore


def run_full_ingestion():
    """Run ingestion on all files"""
    # Get all files
    folder = Path(Config.MARKDOWN_FOLDER)
    files = list(folder.glob('*.md'))
    file_paths = [str(f) for f in files]

    print(f"Found {len(file_paths)} files:")
    for f in file_paths[:5]:
        print(f"   - {Path(f).name}")
    if len(file_paths) > 5:
        print(f"   ... and {len(file_paths) - 5} more")

    # Run ingestion
    vectorstore = ingest_documents(file_paths, drop_old=True)
    return vectorstore


if __name__ == "__main__":
    run_full_ingestion()
