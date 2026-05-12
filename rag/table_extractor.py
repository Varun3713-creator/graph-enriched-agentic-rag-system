"""
rag/table_extractor.py — Extract tables from text and PDF, convert to JSON
"""
from __future__ import annotations
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_tables_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Detect markdown-style or pipe-delimited tables in plain text and convert to JSON records.
    Returns list of {headers, rows, json_data, raw_text}
    """
    tables = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        if _is_table_row(lines[i]):
            table_lines = []
            while i < len(lines) and (lines[i].strip() == "" or _is_table_row(lines[i])):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                table = _parse_table_lines(table_lines)
                if table:
                    tables.append(table)
        else:
            i += 1

    return tables


def _is_table_row(line: str) -> bool:
    """Check if a line looks like a table row."""
    stripped = line.strip()
    return "|" in stripped or re.match(r"^\s*[\w\s]+\s{2,}[\w\s]+", stripped) is not None


def _parse_table_lines(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Parse table lines into structured data."""
    rows = []
    for line in lines:
        # Skip separator lines (e.g., |---|---|)
        if re.match(r"^[\s\|\-\+]+$", line):
            continue
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
        else:
            cells = re.split(r"\s{2,}", line.strip())
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return None

    headers = rows[0]
    data_rows = rows[1:]
    json_data = []
    for row in data_rows:
        record: Dict[str, str] = {}
        for i, header in enumerate(headers):
            record[header] = row[i] if i < len(row) else ""
        json_data.append(record)

    return {
        "headers": headers,
        "rows": data_rows,
        "json_data": json_data,
        "raw_text": "\n".join(lines),
    }


def extract_tables_from_pdf(path: str) -> List[Dict[str, Any]]:
    """Extract tables directly from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping PDF table extraction.")
        return []

    tables = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                headers = [str(h or "").strip() for h in table[0]]
                json_data = []
                for row in table[1:]:
                    record = {headers[i]: str(row[i] or "").strip() for i in range(len(headers)) if i < len(row)}
                    json_data.append(record)
                tables.append({
                    "headers": headers,
                    "rows": [[str(c or "") for c in r] for r in table[1:]],
                    "json_data": json_data,
                    "page": page_num + 1,
                    "raw_text": json.dumps(json_data),
                })
    return tables


def table_to_chunk_text(table: Dict[str, Any]) -> str:
    """Serialize table JSON data as readable text for embedding."""
    return "TABLE DATA:\n" + json.dumps(table["json_data"], indent=2)
