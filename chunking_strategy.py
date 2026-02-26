"""
Chunking Strategy for Normalized Chandra RAG Application

This script implements a hybrid multi-stage chunking approach optimized for:
1. Tabular data (piracy incident reports)
2. Specific queries (ship names, dates, locations)
3. General queries (trends, year-over-year comparisons)
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """Represents a single chunk of text with metadata"""
    content: str
    chunk_id: str
    metadata: Dict[str, Any]
    parent_id: str = None  # ID of the parent document/section
    chunk_type: str = "child"  # "child" or "parent"

    def to_dict(self):
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "chunk_type": self.chunk_type
        }


@dataclass
class Document:
    """Represents a document with its chunks"""
    file_path: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[Chunk] = None

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "metadata": self.metadata,
            "chunks": [c.to_dict() for c in self.chunks] if self.chunks else []
        }


class TableAwareChunker:
    """
    Splits text while keeping markdown tables intact.
    Each table row stays as a single unit.
    """

    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_tables(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into table and non-table sections.
        Returns list of (section_type, content) tuples.
        """
        sections = []

        # Split by markdown tables (| ... | ... |)
        # Table pattern: starts with | and has multiple | in the line
        lines = text.split('\n')
        current_text = []

        for line in lines:
            # Check if line is part of a table (starts with |)
            if line.strip().startswith('|'):
                # Save accumulated text if any
                if current_text:
                    sections.append(('text', '\n'.join(current_text)))
                    current_text = []
                sections.append(('table', line))
            else:
                current_text.append(line)

        # Don't forget remaining text
        if current_text:
            sections.append(('text', '\n'.join(current_text)))

        return sections

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Main chunking method"""
        chunks = []
        sections = self.split_tables(text)

        current_chunk = []
        current_size = 0

        for section_type, content in sections:
            content_size = len(content)

            # If single section exceeds max, we need to handle it
            if content_size > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                        metadata=metadata.copy(),
                        parent_id=doc_id
                    ))
                    current_chunk = []
                    current_size = 0

                # Handle table row that exceeds max size (rare but possible)
                if section_type == 'table':
                    # For tables, we keep them as-is even if large
                    # but add overflow indicator
                    chunks.append(Chunk(
                        content=content,
                        chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                        metadata=metadata.copy(),
                        parent_id=doc_id
                    ))
                else:
                    # Split large text sections
                    sub_chunks = self._split_large_text(content, doc_id, len(chunks), metadata)
                    chunks.extend(sub_chunks)

            # Check if adding this would exceed max
            elif current_size + content_size > self.max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                        metadata=metadata.copy(),
                        parent_id=doc_id
                    ))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(content)
                current_size = content_size
            else:
                current_chunk.append(content)
                current_size += content_size

        # Flush remaining
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(Chunk(
                content=chunk_content,
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                metadata=metadata.copy(),
                parent_id=doc_id
            ))

        return chunks

    def _split_large_text(self, text: str, doc_id: str, start_idx: int, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split large text sections at logical boundaries"""
        chunks = []

        # Try to split at double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        current_para = []
        current_size = 0

        for para in paragraphs:
            if current_size + len(para) > self.max_chunk_size and current_para:
                chunk_content = '\n\n'.join(current_para)
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                    metadata=metadata.copy(),
                    parent_id=doc_id
                ))
                current_para = [para]
                current_size = len(para)
            else:
                current_para.append(para)
                current_size += len(para)

        if current_para:
            chunk_content = '\n\n'.join(current_para)
            chunks.append(Chunk(
                content=chunk_content,
                chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                metadata=metadata.copy(),
                parent_id=doc_id
            ))

        return chunks


class SectionChunker:
    """
    Creates larger parent chunks from sections (e.g., full monthly report).
    Used for Parent Document Retrieval pattern.
    """

    def create_parent_chunks(self, documents: List[Document]) -> List[Chunk]:
        """Create parent chunks from document sections"""
        parent_chunks = []

        for doc in documents:
            # For monthly reports: each document = one parent chunk
            # For annual reports: could split into sections

            report_type = doc.metadata.get('report_type', 'monthly')

            if report_type == 'monthly':
                # Each monthly report is one parent chunk
                parent_chunks.append(Chunk(
                    content=doc.content,
                    chunk_id=f"{doc.metadata['file_name']}_parent",
                    metadata=doc.metadata.copy(),
                    parent_id=None
                ))
            else:
                # For annual reports, split by major sections
                sections = self._split_annual_report(doc.content, doc.metadata)
                parent_chunks.extend(sections)

        return parent_chunks

    def _split_annual_report(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split annual report into logical sections"""
        sections = []

        # Split by ## headings
        parts = re.split(r'(?=^##\s)', content, flags=re.MULTILINE)

        current_section = []
        section_title = "Introduction"

        for part in parts:
            if part.strip().startswith('##'):
                # Save previous section
                if current_section:
                    sections.append(Chunk(
                        content='\n'.join(current_section),
                        chunk_id=f"{metadata['file_name']}_section_{len(sections)}",
                        metadata=metadata.copy(),
                        parent_id=metadata['file_name']
                    ))

                # Extract new section title
                match = re.match(r'^##\s+(.+)$', part.strip(), re.MULTILINE)
                if match:
                    section_title = match.group(1).strip()

                current_section = [part]
            else:
                current_section.append(part)

        # Don't forget last section
        if current_section:
            sections.append(Chunk(
                content='\n'.join(current_section),
                chunk_id=f"{metadata['file_name']}_section_{len(sections)}",
                metadata=metadata.copy(),
                parent_id=metadata['file_name']
            ))

        return sections


class MetadataExtractor:
    """Extract metadata from document content and filename"""

    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    def extract(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract metadata from filename and content"""
        filename = os.path.basename(file_path)

        metadata = {
            'file_path': file_path,
            'file_name': filename,
            'report_type': self._detect_report_type(filename, content),
            'year': self._extract_year(filename, content),
            'month': self._extract_month(filename, content),
        }

        # Extract region/location if present
        if 'MALACCA' in content.upper():
            metadata['regions'] = self._extract_regions(content)

        return metadata

    def _detect_report_type(self, filename: str, content: str) -> str:
        """Detect if monthly or annual report"""
        filename_lower = filename.lower()

        if 'annual' in filename_lower:
            return 'annual'
        elif 'monthly' in filename_lower or 'pirac' in filename_lower:
            return 'monthly'
        else:
            return 'unknown'

    def _extract_year(self, filename: str, content: str) -> int:
        """Extract year from filename or content"""
        # Try filename first
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return int(year_match.group(1))

        # Try content
        year_match = re.search(r'(20\d{2})', content[:500])
        if year_match:
            return int(year_match.group(1))

        return None

    def _extract_month(self, filename: str, content: str) -> str:
        """Extract month from filename"""
        filename_lower = filename.lower()

        for month in self.MONTHS:
            if month.lower() in filename_lower:
                return month

        return None

    def _extract_regions(self, content: str) -> List[str]:
        """Extract regions mentioned in the content"""
        regions = []
        region_keywords = {
            'MALACCA': 'Malacca Strait',
            'SINGAPORE': 'Singapore Strait',
            'WEST AFRICA': 'West Africa',
            'INDIAN OCEAN': 'Indian Ocean',
            'SOUTH CHINA SEA': 'South China Sea',
            'ARABIAN SEA': 'Arabian Sea',
            'GULF OF GUINEA': 'Gulf of Guinea',
        }

        content_upper = content.upper()
        for keyword, region in region_keywords.items():
            if keyword in content_upper:
                regions.append(region)

        return list(set(regions))


class HybridChunker:
    """
    Main orchestrator that combines all chunking strategies.
    Implements the Parent Document Retrieval pattern.
    """

    def __init__(self, child_chunk_size: int = 500, parent_chunk_size: int = 3000):
        self.table_chunker = TableAwareChunker(
            min_chunk_size=200,
            max_chunk_size=child_chunk_size
        )
        self.section_chunker = SectionChunker()
        self.metadata_extractor = MetadataExtractor()

        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size

    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process all documents and return:
        - child_chunks: Small chunks for specific queries
        - parent_chunks: Large chunks for context
        """
        documents = []

        # Step 1: Load and extract metadata
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = self.metadata_extractor.extract(file_path, content)
            documents.append(Document(
                file_path=file_path,
                content=content,
                metadata=metadata
            ))

        # Step 2: Create child chunks (small, specific)
        child_chunks = []
        for doc in documents:
            chunks = self.table_chunker.chunk_text(
                doc.content,
                doc.metadata['file_name'],
                doc.metadata
            )
            child_chunks.extend(chunks)

        # Step 3: Create parent chunks (large, contextual)
        parent_chunks = self.section_chunker.create_parent_chunks(documents)

        # Step 4: Create summaries (optional - for general queries)
        # summaries = self._create_summaries(documents)

        return {
            'documents': documents,
            'child_chunks': child_chunks,
            'parent_chunks': parent_chunks,
            'total_child_chunks': len(child_chunks),
            'total_parent_chunks': len(parent_chunks)
        }


# ============================================================
# DEMONSTRATION
# ============================================================

import json

def demonstrate_chunking():
    """Demonstrate the chunking on test files"""

    # Get test files
    test_dir = Path('test_MD_FILES')
    files = list(test_dir.glob('*.md'))

    print(f"Found {len(files)} test files")
    print("=" * 60)

    # Initialize chunker
    chunker = HybridChunker(child_chunk_size=500, parent_chunk_size=3000)

    # Process documents
    results = chunker.process_documents([str(f) for f in files])

    print(f"\n📊 CHUNKING RESULTS")
    print("=" * 60)
    print(f"Total child chunks: {results['total_child_chunks']}")
    print(f"Total parent chunks: {results['total_parent_chunks']}")

    # Save chunks to JSON file for analysis
    output_data = {
        'summary': {
            'total_documents': len(results['documents']),
            'total_child_chunks': results['total_child_chunks'],
            'total_parent_chunks': results['total_parent_chunks']
        },
        'child_chunks': [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'parent_id': chunk.parent_id,
                'size': len(chunk.content)
            }
            for chunk in results['child_chunks']
        ],
        'parent_chunks': [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'size': len(chunk.content)
            }
            for chunk in results['parent_chunks']
        ]
    }

    output_file = 'RAG_PIPELINE/chunks_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Chunks saved to: {output_file}")

    # Show sample chunks
    print(f"\n📄 SAMPLE CHILD CHUNKS (First 5)")
    print("=" * 60)

    for i, chunk in enumerate(results['child_chunks'][:5]):
        print(f"\n{'='*60}")
        print(f"CHILD CHUNK {i+1}")
        print(f"{'='*60}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Size: {len(chunk.content)} chars")
        print(f"Parent ID: {chunk.parent_id}")
        print(f"Metadata:")
        print(f"  - year: {chunk.metadata.get('year')}")
        print(f"  - month: {chunk.metadata.get('month')}")
        print(f"  - report_type: {chunk.metadata.get('report_type')}")
        print(f"  - regions: {chunk.metadata.get('regions', [])[:3]}...")
        print(f"\nContent:\n{chunk.content[:600]}")
        if len(chunk.content) > 600:
            print(f"\n... [truncated to 600 chars, total: {len(chunk.content)}]")

    print(f"\n\n📑 SAMPLE PARENT CHUNKS (First 3)")
    print("=" * 60)

    for i, chunk in enumerate(results['parent_chunks'][:3]):
        print(f"\n{'='*60}")
        print(f"PARENT CHUNK {i+1}")
        print(f"{'='*60}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Size: {len(chunk.content)} chars")
        print(f"Metadata:")
        print(f"  - year: {chunk.metadata.get('year')}")
        print(f"  - month: {chunk.metadata.get('month')}")
        print(f"  - report_type: {chunk.metadata.get('report_type')}")
        print(f"  - regions: {chunk.metadata.get('regions', [])[:3]}...")
        print(f"\nContent preview:\n{chunk.content[:800]}")
        if len(chunk.content) > 800:
            print(f"\n... [truncated to 800 chars, total: {len(chunk.content)}]")

    print(f"\n\n✅ All {results['total_child_chunks']} child chunks and {results['total_parent_chunks']} parent chunks saved to JSON")

    return results


if __name__ == "__main__":
    results = demonstrate_chunking()
