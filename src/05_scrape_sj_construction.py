"""
scrape_sj_construction.py
--------------------------
Scrapes the San Jose new construction report listing page, downloads
every monthly PDF report, extracts the permit table from each, and
saves all records to a single CSV.

Usage:
    pip install beautifulsoup4 pdfplumber pandas
    python scrape_sj_construction.py

Output:
    sj_construction_reports.csv  — one row per permit across all reports
"""

import subprocess
import tempfile
import os
import re
import time
import io

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_URL    = "https://csjpbce.sanjoseca.gov"
LISTING_URL = "https://csjpbce.sanjoseca.gov/reportviewer/getreports.asp?rt=nc"
OUTPUT_CSV  = "sj_construction_reports.csv"
DELAY       = 0.5   # seconds between requests

# Pattern that matches the monthly construction report PDFs
PDF_PATTERN = re.compile(r"newconstruction/nc_\d{4}-\d{2}_\w+\.pdf", re.IGNORECASE)


# ── FETCH URL (curl-based, bypasses TLS fingerprinting) ───────────────────────

def fetch_bytes(url):
    """Fetches a URL via curl and returns raw bytes, or None on failure."""
    result = subprocess.run(
        [
            "curl", "-s", "-L",
            "--max-time", "60",
            "--tlsv1.2",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "application/pdf,*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.9",
            "-H", "Connection: keep-alive",
            url,
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        print(f"  curl failed for {url}: {result.stderr[:120]}")
        return None
    return result.stdout


def fetch_html(url):
    """Fetches a URL and returns decoded HTML text."""
    raw = fetch_bytes(url)
    if raw is None:
        return None
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


# ── STEP 1: COLLECT PDF LINKS FROM LISTING PAGE ───────────────────────────────

def get_report_links():
    print("Fetching listing page...")
    html = fetch_html(LISTING_URL)
    if not html:
        raise RuntimeError("Could not fetch listing page.")

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].replace("\\", "/")  # normalize any backslashes
        if PDF_PATTERN.search(href):
            full_url = urljoin(BASE_URL, href)
            # Extract year-month from filename, e.g. nc_2004-12_Dec2004.pdf → 2004-12
            match = re.search(r"nc_(\d{4}-\d{2})_", href, re.IGNORECASE)
            label = match.group(1) if match else "unknown"
            links.append({"url": full_url, "report_month": label})

    print(f"  Found {len(links)} construction report PDFs.")
    return links


# ── STEP 2: EXTRACT TABLE DATA FROM A SINGLE PDF ──────────────────────────────

def extract_pdf_tables(pdf_bytes, report_month):
    """
    Opens a PDF from bytes and extracts all table rows.
    Returns a list of dicts (one per data row).
    """
    rows = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            headers = None
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # Determine if first row looks like a header
                    first_row = [str(c).strip() if c else "" for c in table[0]]
                    # Use first row as header if no headers yet, or if it
                    # looks like a repeat header (contains permit-like keywords)
                    is_header_row = any(
                        kw in " ".join(first_row).lower()
                        for kw in ("permit", "address", "owner", "type",
                                   "value", "date", "description", "contractor",
                                   "no", "number", "status")
                    )
                    if headers is None or is_header_row:
                        headers = first_row
                        data_rows = table[1:]
                    else:
                        data_rows = table

                    for row in data_rows:
                        if not any(cell for cell in row):
                            continue  # skip blank rows
                        cleaned = [str(c).strip() if c else "" for c in row]
                        # Pad or trim to match header length
                        while len(cleaned) < len(headers):
                            cleaned.append("")
                        cleaned = cleaned[:len(headers)]
                        record = dict(zip(headers, cleaned))
                        record["report_month"] = report_month
                        rows.append(record)

    except Exception as e:
        print(f"    pdfplumber error: {e}")

    return rows


# ── STEP 3: FALLBACK — raw text extraction if no tables found ─────────────────

def extract_pdf_text_lines(pdf_bytes, report_month):
    """
    Fallback: extracts raw text lines as rows when no structured tables exist.
    Each line becomes one row with a single 'text' column.
    """
    rows = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        rows.append({"text": line, "report_month": report_month})
    except Exception as e:
        print(f"    text fallback error: {e}")
    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

links = get_report_links()

if not links:
    print("No PDF links found. Check LISTING_URL or PDF_PATTERN.")
    raise SystemExit(1)

all_records = []
text_fallback_reports = []

for i, link in enumerate(links, 1):
    url          = link["url"]
    report_month = link["report_month"]
    print(f"  [{i:3d}/{len(links)}] {report_month}  {url.split('/')[-1]}")

    pdf_bytes = fetch_bytes(url)
    if pdf_bytes is None:
        print("    Skipping — download failed.")
        continue

    rows = extract_pdf_tables(pdf_bytes, report_month)

    if rows:
        all_records.extend(rows)
        print(f"    → {len(rows)} rows extracted (table mode)")
    else:
        # Fall back to raw text
        rows = extract_pdf_text_lines(pdf_bytes, report_month)
        if rows:
            all_records.extend(rows)
            text_fallback_reports.append(report_month)
            print(f"    → {len(rows)} lines extracted (text fallback)")
        else:
            print("    → No data extracted.")

    time.sleep(DELAY)

# ── BUILD DATAFRAME AND SAVE ──────────────────────────────────────────────────

if not all_records:
    print("\nNo data was collected across all PDFs.")
    raise SystemExit(1)

df = pd.DataFrame(all_records)

# Move report_month to first column
cols = ["report_month"] + [c for c in df.columns if c != "report_month"]
df = df[cols]

# Drop entirely empty columns
df = df.dropna(axis=1, how="all")
df = df.replace("", pd.NA)
df = df.dropna(how="all")  # drop rows where every field is blank

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}")
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Reports    : {df['report_month'].nunique()} months")
if text_fallback_reports:
    print(f"  Text-only fallback used for: {text_fallback_reports}")
