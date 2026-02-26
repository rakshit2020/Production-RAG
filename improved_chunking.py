"""
Improved Chunking Strategy for Normalized Chandra RAG Application

FIXED ISSUES:
- Tables are kept intact (no row splitting)
- Chunks split at logical boundaries (headings, sections)
- Proper context preserved in each chunk
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a single chunk of text with metadata"""
    content: str
    chunk_id: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'section', 'incident', 'table_row'
    parent_id: str = None  # ID of the parent document/section


@dataclass
class Document:
    """Represents a document"""
    file_path: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[Chunk] = None


class LogicalSectionChunker:
    """
    Chunks by logical sections rather than character limits.
    Keeps table rows and incidents intact.
    """

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def chunk_document(self, content: str, doc_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Main chunking method using logical boundaries"""
        chunks = []
        chunk_counter = 0

        # Split content into logical sections
        sections = self._split_into_sections(content)

        for section in sections:
            section_type = section['type']
            section_content = section['content']
            section_title = section.get('title', '')

            if section_type == 'heading':
                # For headings, just add as small chunk (will be merged)
                chunk = Chunk(
                    content=section_content,
                    chunk_id=f"{doc_id}_chunk_{chunk_counter}",
                    metadata=metadata.copy(),
                    chunk_type='heading'
                )
                chunks.append(chunk)
                chunk_counter += 1

            elif section_type == 'table':
                # For tables, process row by row
                table_chunks = self._chunk_table(section_content, doc_id, metadata, chunk_counter)
                chunks.extend(table_chunks)
                chunk_counter += len(table_chunks)

            elif section_type == 'text':
                # For text paragraphs, split at logical points
                text_chunks = self._chunk_text(section_content, doc_id, metadata, chunk_counter)
                chunks.extend(text_chunks)
                chunk_counter += len(text_chunks)

            elif section_type == 'mixed':
                # Mixed content - process each part
                mixed_chunks = self._chunk_mixed_section(section_content, doc_id, metadata, chunk_counter)
                chunks.extend(mixed_chunks)
                chunk_counter += len(mixed_chunks)

        # Merge small chunks with neighbors to ensure minimum size
        chunks = self._merge_small_chunks(chunks, min_size=300)

        return chunks

    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split document into logical sections"""
        sections = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for main heading (##)
            if line.startswith('##'):
                sections.append({
                    'type': 'heading',
                    'title': line.strip('# '),
                    'content': line
                })
                i += 1

            # Check for section headers like "IN TERRITORIAL WATERS" or "IN PORT AREA"
            elif line.startswith('**') and ('TERRITORIAL' in line.upper() or 'PORT' in line.upper() or 'WATERS' in line.upper()):
                sections.append({
                    'type': 'heading',
                    'title': line.strip('* '),
                    'content': line
                })
                i += 1

            # Check for table (starts with |)
            elif line.strip().startswith('|'):
                # Collect entire table
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1

                # Skip separator lines (|---|)
                if table_lines and not self._is_separator_line(table_lines[0]):
                    sections.append({
                        'type': 'table',
                        'title': 'table',
                        'content': '\n'.join(table_lines)
                    })
                continue

            # Regular text
            else:
                # Collect paragraph
                para_lines = []
                while i < len(lines) and not lines[i].strip().startswith('|') and not lines[i].startswith('##') and not (lines[i].startswith('**') and ('TERRITORIAL' in lines[i].upper() or 'PORT' in lines[i].upper())):
                    if lines[i].strip():
                        para_lines.append(lines[i])
                    i += 1

                if para_lines:
                    sections.append({
                        'type': 'text',
                        'title': 'text',
                        'content': '\n'.join(para_lines)
                    })
                continue

        return sections

    def _is_separator_line(self, line: str) -> bool:
        """Check if line is a markdown table separator"""
        return bool(re.match(r'^\|\s*[-:]+\s*\|', line))

    def _chunk_table(self, table_content: str, doc_id: str, metadata: Dict[str, Any], start_idx: int) -> List[Chunk]:
        """Chunk table content - keep rows intact"""
        chunks = []
        lines = table_content.split('\n')

        current_chunk_rows = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            # Skip separator lines
            if self._is_separator_line(line):
                continue

            # Skip header row (contains column descriptions)
            if 'N°' in line or 'Nº' in line:
                continue

            # Skip empty rows
            if not line.strip() or line.strip() == '|':
                continue

            # Check if this is an incident row (starts with a number or letter)
            if self._is_incident_row(line):
                row_size = len(line)

                # If adding this row exceeds limit, flush current chunk
                if current_size + row_size > self.max_chunk_size and current_chunk_rows:
                    chunk_content = '\n'.join(current_chunk_rows)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                        metadata=metadata.copy(),
                        chunk_type='incident_table'
                    ))
                    current_chunk_rows = []
                    current_size = 0

                current_chunk_rows.append(line)
                current_size += row_size
            else:
                # Non-incident table row (like section headers within table)
                if current_chunk_rows:
                    chunk_content = '\n'.join(current_chunk_rows)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                        metadata=metadata.copy(),
                        chunk_type='incident_table'
                    ))
                    current_chunk_rows = []
                    current_size = 0

                chunks.append(Chunk(
                    content=line,
                    chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                    metadata=metadata.copy(),
                    chunk_type='table_row'
                ))

        # Don't forget last chunk
        if current_chunk_rows:
            chunk_content = '\n'.join(current_chunk_rows)
            chunks.append(Chunk(
                content=chunk_content,
                chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                metadata=metadata.copy(),
                chunk_type='incident_table'
            ))

        return chunks

    def _is_incident_row(self, line: str) -> bool:
        """Check if table row is an incident (starts with number or 'IN PORT')"""
        # Remove leading | and whitespace
        stripped = line.lstrip('|').strip()

        # Check if starts with a number
        if stripped and stripped[0].isdigit():
            return True

        # Check if it's a section header like "IN PORT AREA"
        if 'IN PORT' in stripped.upper() or 'TERRITORIAL' in stripped.upper():
            return True

        return False

    def _chunk_text(self, text_content: str, doc_id: str, metadata: Dict[str, Any], start_idx: int) -> List[Chunk]:
        """Chunk text content at paragraph boundaries"""
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text_content)

        current_para = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If single paragraph is huge, split it further
            if len(para) > self.max_chunk_size:
                # Flush current
                if current_para:
                    chunk_content = '\n\n'.join(current_para)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                        metadata=metadata.copy(),
                        chunk_type='text'
                    ))
                    current_para = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_size + len(sent) > self.max_chunk_size and current_para:
                        chunk_content = '\n\n'.join(current_para)
                        chunks.append(Chunk(
                            content=chunk_content,
                            chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                            metadata=metadata.copy(),
                            chunk_type='text'
                        ))
                        current_para = []
                        current_size = 0
                    current_para.append(sent)
                    current_size += len(sent)
            else:
                if current_size + len(para) > self.max_chunk_size and current_para:
                    chunk_content = '\n\n'.join(current_para)
                    chunks.append(Chunk(
                        content=chunk_content,
                        chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                        metadata=metadata.copy(),
                        chunk_type='text'
                    ))
                    current_para = []
                    current_size = 0

                current_para.append(para)
                current_size += len(para)

        # Flush remaining
        if current_para:
            chunk_content = '\n\n'.join(current_para)
            chunks.append(Chunk(
                content=chunk_content,
                chunk_id=f"{doc_id}_chunk_{start_idx + len(chunks)}",
                metadata=metadata.copy(),
                chunk_type='text'
            ))

        return chunks

    def _chunk_mixed_section(self, section_content: str, doc_id: str, metadata: Dict[str, Any], start_idx: int) -> List[Chunk]:
        """Handle mixed content (text + tables together)"""
        chunks = []
        lines = section_content.split('\n')

        current_text = []
        chunk_counter = 0

        for line in lines:
            if line.strip().startswith('|'):
                # Flush text if any
                if current_text:
                    text_content = '\n'.join(current_text)
                    text_chunks = self._chunk_text(text_content, doc_id, metadata, start_idx + chunk_counter)
                    chunks.extend(text_chunks)
                    chunk_counter += len(text_chunks)
                    current_text = []

                # Process table line with surrounding context
                chunks.append(Chunk(
                    content=line,
                    chunk_id=f"{doc_id}_chunk_{start_idx + chunk_counter}",
                    metadata=metadata.copy(),
                    chunk_type='incident_table'
                ))
                chunk_counter += 1
            else:
                current_text.append(line)

        # Don't forget text
        if current_text:
            text_content = '\n'.join(current_text)
            text_chunks = self._chunk_text(text_content, doc_id, metadata, start_idx + chunk_counter)
            chunks.extend(text_chunks)

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk], min_size: int = 300) -> List[Chunk]:
        """Merge small chunks with neighbors to ensure meaningful size"""
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If current chunk is too small, try to merge with next
            if len(current.content) < min_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]

                # Merge with next if they're compatible types
                merged_content = current.content + '\n\n' + next_chunk.content

                # Only merge if result is reasonable
                if len(merged_content) <= self.max_chunk_size * 1.5:
                    merged.append(Chunk(
                        content=merged_content,
                        chunk_id=current.chunk_id,
                        metadata=current.metadata,
                        chunk_type='merged'
                    ))
                    i += 2
                    continue

            # If can't merge, keep as is
            merged.append(current)
            i += 1

        return merged


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

        if 'MALACCA' in content.upper():
            metadata['regions'] = self._extract_regions(content)

        return metadata

    def _detect_report_type(self, filename: str, content: str) -> str:
        filename_lower = filename.lower()
        if 'annual' in filename_lower:
            return 'annual'
        elif 'monthly' in filename_lower or 'pirac' in filename_lower:
            return 'monthly'
        return 'unknown'

    def _extract_year(self, filename: str, content: str) -> int:
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return int(year_match.group(1))
        year_match = re.search(r'(20\d{2})', content[:500])
        if year_match:
            return int(year_match.group(1))
        return None

    def _extract_month(self, filename: str, content: str) -> str:
        filename_lower = filename.lower()
        for month in self.MONTHS:
            if month.lower() in filename_lower:
                return month
        return None

    def _extract_regions(self, content: str) -> List[str]:
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


class ImprovedChunker:
    """Main chunker with logical boundaries + Parent Document Retrieval"""

    def __init__(self, max_chunk_size: int = 2000):
        self.section_chunker = LogicalSectionChunker(max_chunk_size=max_chunk_size)
        self.metadata_extractor = MetadataExtractor()
        self.max_chunk_size = max_chunk_size

    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process all documents with parent-child hierarchy"""
        documents = []
        child_chunks = []
        parent_chunks = []

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = self.metadata_extractor.extract(file_path, content)
            doc_id = metadata['file_name']

            # Create PARENT chunk (full document)
            parent_chunk = Chunk(
                content=content,
                chunk_id=f"{doc_id}_parent",
                metadata=metadata.copy(),
                chunk_type='parent_document',
                parent_id=None
            )
            parent_chunks.append(parent_chunk)

            # Create CHILD chunks (logical sections)
            childs = self.section_chunker.chunk_document(content, doc_id, metadata)

            # Link each child to parent
            for child in childs:
                child.parent_id = f"{doc_id}_parent"

            child_chunks.extend(childs)

            documents.append(Document(
                file_path=file_path,
                content=content,
                metadata=metadata,
                chunks=childs
            ))

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

def demonstrate_improved_chunking():
    """Demonstrate improved chunking"""

    test_dir = Path('test_MD_FILES')
    files = list(test_dir.glob('*.md'))

    print(f"Found {len(files)} test files")
    print("=" * 60)

    chunker = ImprovedChunker(max_chunk_size=2000)
    results = chunker.process_documents([str(f) for f in files])

    print(f"\n📊 CHUNKING RESULTS")
    print("=" * 60)
    print(f"Total child chunks: {results['total_child_chunks']}")
    print(f"Total parent chunks: {results['total_parent_chunks']}")

    # Analyze chunk types
    chunk_types = {}
    for chunk in results['child_chunks']:
        t = chunk.chunk_type
        chunk_types[t] = chunk_types.get(t, 0) + 1

    print(f"\nChild chunk types breakdown:")
    for t, count in chunk_types.items():
        print(f"  - {t}: {count}")

    # Save to JSON
    output_data = {
        'summary': {
            'total_documents': len(results['documents']),
            'total_child_chunks': results['total_child_chunks'],
            'total_parent_chunks': results['total_parent_chunks'],
            'chunk_types': chunk_types
        },
        'child_chunks': [
            {
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'content': chunk.content,
                'parent_id': chunk.parent_id,
                'metadata': {
                    'year': chunk.metadata.get('year'),
                    'month': chunk.metadata.get('month'),
                    'report_type': chunk.metadata.get('report_type')
                },
                'size': len(chunk.content)
            }
            for chunk in results['child_chunks']
        ],
        'parent_chunks': [
            {
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'content': chunk.content,
                'metadata': {
                    'year': chunk.metadata.get('year'),
                    'month': chunk.metadata.get('month'),
                    'report_type': chunk.metadata.get('report_type')
                },
                'size': len(chunk.content)
            }
            for chunk in results['parent_chunks']
        ]
    }

    output_file = 'RAG_PIPELINE/chunks_output_improved.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {output_file}")

    # Show sample chunks
    print(f"\n📄 SAMPLE CHUNKS")
    print("=" * 60)

    # Show incident table chunks specifically
    incident_chunks = [c for c in results['child_chunks'] if c.chunk_type == 'incident_table']

    print(f"\n--- Sample INCIDENT TABLE chunks (complete rows) ---")
    for i, chunk in enumerate(incident_chunks[:3]):
        print(f"\n{'='*60}")
        print(f"CHUNK {i+1} ({chunk.chunk_type})")
        print(f"Parent ID: {chunk.parent_id}")
        print(f"{'='*60}")
        print(f"Size: {len(chunk.content)} chars")
        print(f"Metadata: year={chunk.metadata.get('year')}, month={chunk.metadata.get('month')}")
        print(f"\nContent:")
        print(chunk.content[:500])

    # Show parent chunks
    print(f"\n\n--- Sample PARENT chunks (full documents) ---")
    for i, chunk in enumerate(results['parent_chunks'][:2]):
        print(f"\n{'='*60}")
        print(f"PARENT {i+1} ({chunk.chunk_type})")
        print(f"{'='*60}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Size: {len(chunk.content)} chars")
        print(f"Content preview:")
        print(chunk.content[:300])

    return results


if __name__ == "__main__":
    results = demonstrate_improved_chunking()
