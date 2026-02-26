import re
import uuid
import pandas as pd
from pathlib import Path
from dateutil import parser as dateparser

from langchain_core.documents import Document


# =========================================================
# Utilities
# =========================================================

def generate_id():
    return str(uuid.uuid4())


def extract_year_from_text(text):
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))
    return None


def safe_parse_date(text):
    try:
        return dateparser.parse(text, fuzzy=True)
    except:
        return None


def normalize_text(x):
    """
    Robust normalization for scalar / Series / list values.
    """
    try:
        if hasattr(x, "tolist"):
            x = " ".join([str(i) for i in x.tolist() if str(i) != "nan"])

        if pd.isna(x):
            return ""

        return str(x).strip()

    except Exception:
        return str(x).strip()


def extract_ship_name(text):
    """
    Very robust ship name guess from row text.
    """
    if not text:
        return ""

    words = text.split()

    if len(words) == 0:
        return ""

    # Usually first token is ship name
    return words[0]


# =========================================================
# Markdown Parsing
# =========================================================

def split_markdown_sections(md_text):
    pattern = r"(#+ .+)"
    parts = re.split(pattern, md_text)

    sections = []
    current_heading = None

    for part in parts:
        if part.startswith("#"):
            current_heading = part.strip()
        else:
            if current_heading:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": part.strip()
                    }
                )

    return sections


def extract_tables_from_section(section_text):
    tables = []

    lines = section_text.split("\n")
    table_buffer = []

    for line in lines:
        if "|" in line:
            table_buffer.append(line)
        else:
            if table_buffer:
                tables.append("\n".join(table_buffer))
                table_buffer = []

    if table_buffer:
        tables.append("\n".join(table_buffer))

    return tables


def markdown_table_to_dataframe(table_text):
    lines = [l.strip() for l in table_text.split("\n") if l.strip()]

    if len(lines) < 2:
        return None

    header = lines[0]
    separator = lines[1]

    if "---" not in separator:
        return None

    data_lines = lines[2:]

    raw_cols = [c.strip() for c in header.split("|") if c.strip()]

    # Make columns unique
    columns = []
    seen = {}

    for col in raw_cols:
        if col in seen:
            seen[col] += 1
            columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            columns.append(col)

    rows = []

    for line in data_lines:
        row = [c.strip() for c in line.split("|") if c.strip()]

        if len(row) == len(columns):
            rows.append(row)
        else:
            # fallback: treat entire line as single column
            rows.append([line.strip()])

    try:
        df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
    except:
        df = pd.DataFrame(rows)

    df = df.fillna("")

    return df


# =========================================================
# Chunk Builders
# =========================================================

def build_incident_chunks(df, heading, source_file, report_year):
    documents = []

    if df is None or df.shape[0] == 0:
        return documents

    for idx, row in df.iterrows():

        row_dict = {str(col): normalize_text(row[col]) for col in df.columns}

        row_text = " ".join(row_dict.values())

        if len(row_text.strip()) < 10:
            continue

        # Ship name extraction
        ship_name = extract_ship_name(row_text)

        # Date extraction
        parsed_date = safe_parse_date(row_text)

        incident_date = parsed_date.date().isoformat() if parsed_date else None
        year = parsed_date.year if parsed_date else report_year

        incident_id = f"{year}-{idx}-{uuid.uuid4().hex[:6]}"

        # Build structured content
        content_lines = [
            f"Incident ID: {incident_id}",
            f"Ship Name: {ship_name}",
        ]

        for k, v in row_dict.items():
            if v:
                content_lines.append(f"{k}: {v}")

        content_lines.append(f"Section: {heading}")

        page_content = "\n".join(content_lines)

        metadata = {
            "chunk_type": "incident",
            "incident_id": incident_id,
            "ship_name": ship_name.lower(),
            "year": year,
            "incident_date": incident_date,
            "section": heading,
            "source_file": source_file,
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return documents


def build_section_chunk(heading, text, source_file, year):
    return Document(
        page_content=f"{heading}\n\n{text}",
        metadata={
            "chunk_type": "section",
            "heading": heading,
            "year": year,
            "source_file": source_file,
        }
    )


def build_report_chunk(text, source_file, year):
    return Document(
        page_content=text,
        metadata={
            "chunk_type": "report",
            "year": year,
            "source_file": source_file,
        }
    )


# =========================================================
# Main Chunking Pipeline
# =========================================================

def chunk_markdown_file(file_path):
    file_path = Path(file_path)
    source_file = file_path.name

    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    report_year = extract_year_from_text(md_text)

    documents = []

    sections = split_markdown_sections(md_text)

    for sec in sections:
        heading = sec["heading"]
        content = sec["content"]

        tables = extract_tables_from_section(content)

        # ---------- Incident chunks ----------
        if tables:
            for table_text in tables:
                df = markdown_table_to_dataframe(table_text)

                docs = build_incident_chunks(
                    df,
                    heading,
                    source_file,
                    report_year
                )

                documents.extend(docs)

        # ---------- Section chunk ----------
        if len(content.strip()) > 80:
            documents.append(
                build_section_chunk(
                    heading,
                    content,
                    source_file,
                    report_year
                )
            )

    # ---------- Report chunk ----------
    documents.append(
        build_report_chunk(
            md_text[:2000],
            source_file,
            report_year
        )
    )

    return documents


# =========================================================
# Batch Processing
# =========================================================

def chunk_folder(folder_path):
    all_docs = []

    folder = Path(folder_path)

    for file in folder.glob("*.md"):
        print(f"Processing: {file.name}")

        docs = chunk_markdown_file(file)

        print(f"  Chunks Created: {len(docs)}")

        all_docs.extend(docs)

    return all_docs


# =========================================================
# Debug / Test Runner
# =========================================================

if __name__ == "__main__":

    folder_path = "/home/rakshit/Desktop/COOKING/PROJECTS/LangChain_Setup/RAG_scripts_CAIR/test_MD_FILES"   # change to your folder

    docs = chunk_folder(folder_path)

    print("\n============================")
    print(f"Total Chunks: {len(docs)}")

    type_count = {}

    for d in docs:
        t = d.metadata["chunk_type"]
        type_count[t] = type_count.get(t, 0) + 1

    print("\nChunk Types Distribution:")
    for k, v in type_count.items():
        print(f"{k}: {v}")

    print("\n============================")
    print("Sample Chunk:\n")

    for i, doc in enumerate(docs[:10]):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:800])
        print("\nMetadata:\n", doc.metadata)
        print("-" * 40)
