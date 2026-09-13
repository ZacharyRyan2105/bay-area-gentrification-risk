"""
parse_sj_construction.py
------------------------
Reads sj_construction_reports.csv (the raw text-line output from the scraper)
and restructures it so each row is one permit.

Input:   sj_construction_reports.csv
Output:  sj_permits_clean.csv

Each permit spans 3 text lines in the PDF:
  Line 1: FOLDER_NUMBER APPLICANT APPROVALS VALUATION SQ_FT
  Line 2: [LOT] PERMIT_DATE CONTRACTOR SUB_TYPE [CENSUS_CODE] DWELL_UNITS
  Line 3: ADDRESS (WORK_CODE%) DESCRIPTION WORK_TYPE

Usage:
    python parse_sj_construction.py
"""

import pandas as pd
import re

INPUT_CSV  = "sj_construction_reports.csv"
OUTPUT_CSV = "sj_permits_clean.csv"

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────

# Line 1: optional leading tract number before folder number
FOLDER_RE    = re.compile(r'^(?:\d+\s+)?(\d{4}-\d{6}-\d{3}-\d{2}-[A-Z]{2})\s+(.*)', re.DOTALL)
# Line 2: optional leading lot number before date
DATE_RE      = re.compile(r'^(?:\d+\s+)?(\d{2}/\d{2}/\d{4})\s+(.*)', re.DOTALL)
VALUATION_RE = re.compile(r'([\d,]+\.\d{2})\s+(\d+)\s*$')
APPROVALS_RE = re.compile(r'([A-Z]-\w+(?:\s+only)?(?:,\s*[A-Z]-\w+(?:\s+only)?)*)')
WORK_PAREN_RE= re.compile(r'\(([A-Z]+)\s+(\d+)%\)')
WORK_TYPE_RE = re.compile(
    r'(New Construction|Finish Interior|Addition|Demolition|Repair|'
    r'Change of Use|Alteration|Tenant Improvement|Re-Roof|Pool|Grading|'
    r'Foundation Only|Shell Only)'
    r'[\w\s]{0,20}$',    # allow optional trailing codes/numbers (district, permit ref)
    re.IGNORECASE)

# Loose fallback: catches garbled OCR like "CONSTRUCTNIOewN Construction"
WORK_TYPE_FUZZY = [
    (re.compile(r'[Nn]ew.{0,3}[Cc]onstruct', re.IGNORECASE), 'New Construction'),
    (re.compile(r'[Ff]inish.{0,3}[Ii]nterior', re.IGNORECASE), 'Finish Interior'),
    (re.compile(r'[Aa]ddition', re.IGNORECASE),                 'Addition'),
    (re.compile(r'[Dd]emolition', re.IGNORECASE),               'Demolition'),
    (re.compile(r'[Aa]lteration', re.IGNORECASE),               'Alteration'),
    (re.compile(r'[Tt]enant\s*[Ii]mprovement', re.IGNORECASE), 'Tenant Improvement'),
    (re.compile(r'[Cc]hange\s*of\s*[Uu]se', re.IGNORECASE),   'Change of Use'),
]

# Lines to skip (headers / footers / blank)
SKIP_RE = re.compile(
    r'City of San Jose|PBCE Department|Date : From|'
    r'Tract Folder|Lot Permit|APN Job|Printed Date|^\s*$',
    re.IGNORECASE)

# Known building sub-types (longest match wins)
KNOWN_SUBTYPES = sorted([
    'Single Family', 'Condo', 'Apartment', 'Apartment/Condo', 'Office',
    'Canopy Buildings', 'Commercial', 'Industrial', 'Mixed Use', 'Hotel',
    'Motel', 'Church', 'School', 'Hospital', 'Garage', 'Carport',
    'Duplex', 'Townhouse', 'Retail', 'Restaurant', 'Warehouse', 'Storage',
    '2nd Unit Added', 'Shell Building', 'Tenant Improvement',
    'Medical/Dental Clinic', 'Medical/Dental', 'Manufacturing', 'Assembly',
    'Bank', 'Theater', 'Undefined', 'Auto Dealer', 'Service Station',
    'Day Care', 'Library', 'Park/Recreation', 'Utility', 'Parking',
    'Auto Repair', 'Car Wash', 'Gas Station', 'Fitness', 'Salon',
], key=len, reverse=True)


# ── LINE PARSERS ──────────────────────────────────────────────────────────────

def parse_line1(text):
    m = FOLDER_RE.match(text)
    if not m:
        return None
    folder, rest = m.group(1), m.group(2)

    vm = VALUATION_RE.search(rest)
    if vm:
        valuation = float(vm.group(1).replace(',', ''))
        sq_ft     = int(vm.group(2))
        middle    = rest[:vm.start()].strip()
    else:
        # Overflowed line — valuation missing, keep what we have
        valuation = None
        sq_ft     = None
        middle    = rest.strip()

    ap_m = APPROVALS_RE.search(middle)
    if ap_m:
        approvals = ap_m.group(1)
        applicant = middle[:ap_m.start()].strip()
    else:
        approvals = ''
        applicant = middle

    return dict(folder_number=folder, applicant=applicant,
                approvals=approvals, valuation=valuation, sq_ft=sq_ft)


def parse_line2(text):
    m = DATE_RE.match(text)
    if not m:
        return None
    date, rest = m.group(1), m.group(2)

    # Dwell units = last integer; census code = second-to-last if also integer
    tokens = rest.split()
    dwell = census = None
    if tokens and tokens[-1].isdigit():
        dwell = int(tokens[-1])
        rest2 = rest[:rest.rfind(tokens[-1])].rstrip()
        t2 = rest2.split()
        if t2 and t2[-1].isdigit():
            census = int(t2[-1])
            rest2 = rest2[:rest2.rfind(t2[-1])].rstrip()
    else:
        rest2 = rest

    sub_type   = ''
    contractor = rest2
    for st in KNOWN_SUBTYPES:
        idx = rest2.lower().find(st.lower())
        if idx != -1:
            sub_type   = st
            contractor = rest2[:idx].strip()
            break

    return dict(permit_date=date, contractor=contractor,
                sub_type=sub_type, census_code=census, dwell_units=dwell)


APN_RE = re.compile(r'^\d{8,11}\s+')

def parse_line3(text):
    # Strip leading APN (8-11 digit parcel number)
    text = APN_RE.sub('', text).strip()

    wm        = WORK_TYPE_RE.search(text)
    if wm:
        work_type = wm.group(1).strip()
        body      = text[:wm.start()].strip()
    else:
        # Fuzzy fallback for garbled OCR
        work_type = ''
        body      = text
        for pattern, label in WORK_TYPE_FUZZY:
            fm = pattern.search(text)
            if fm:
                work_type = label
                body      = text[:fm.start()].strip()
                break

    paren_idx   = body.find('(')
    address     = body[:paren_idx].strip() if paren_idx != -1 else body
    wp          = WORK_PAREN_RE.findall(body)
    work_code   = wp[0][0] if wp else ''
    description = body[paren_idx:].strip() if paren_idx != -1 else ''

    return dict(address=address, description=description,
                work_code=work_code, work_type=work_type)


# ── MAIN PARSE LOOP ───────────────────────────────────────────────────────────

print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} raw lines across {df['report_month'].nunique()} monthly reports")

records    = []
skipped    = 0

for month, grp in df.groupby('report_month', sort=True):
    lines = [str(l) for l in grp['text'].tolist() if not SKIP_RE.search(str(l))]
    i = 0
    while i < len(lines):
        l1 = lines[i]
        p1 = parse_line1(l1)
        if p1 is None:
            i += 1
            continue
        if i + 2 >= len(lines):
            break

        l2 = lines[i + 1]
        p2 = parse_line2(l2)
        if p2 is None:
            # Maybe the overflowed portion of line 1 landed on i+1 and date is on i+2
            p2_candidate = parse_line2(lines[i + 2]) if i + 2 < len(lines) else None
            if p2_candidate is not None:
                p2 = p2_candidate
                l3 = lines[i + 3] if i + 3 < len(lines) else ''
                offset = 4
            else:
                skipped += 1
                i += 1
                continue
        else:
            l3 = lines[i + 2]
            offset = 3

        if p2 is None:
            skipped += 1
            i += 1
            continue

        p3 = parse_line3(l3)
        rec = {'report_month': month}
        rec.update(p1)
        rec.update(p2)
        rec.update(p3)
        records.append(rec)
        i += offset

# ── CLEAN AND SAVE ────────────────────────────────────────────────────────────

out = pd.DataFrame(records)

# Reorder columns logically
col_order = [
    'report_month', 'permit_date',
    'folder_number', 'sub_type', 'work_type', 'work_code',
    'applicant', 'contractor',
    'address', 'description',
    'valuation', 'sq_ft', 'dwell_units',
    'approvals', 'census_code',
]
out = out[[c for c in col_order if c in out.columns]]

# Parse permit_date as datetime for sorting
out['permit_date'] = pd.to_datetime(out['permit_date'], format='%m/%d/%Y', errors='coerce')
out = out.sort_values(['permit_date', 'folder_number']).reset_index(drop=True)

# Normalize work_type and sub_type casing
WORK_TYPE_NORM = {
    'new construction':     'New Construction',
    'finish interior':      'Finish Interior',
    'foundation only':      'Foundation Only',
    'tenant improvement':   'Tenant Improvement',
    'change of use':        'Change of Use',
    'shell only':           'Shell Only',
}
out['work_type'] = (out['work_type'].str.strip()
                    .str.lower()
                    .map(lambda x: WORK_TYPE_NORM.get(x, x.title() if x else ''))
                    )

out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {OUTPUT_CSV}")
print(f"  Permits parsed : {len(out):,}")
print(f"  Skipped lines  : {skipped}")
print(f"  Date range     : {out['permit_date'].min().date()} → {out['permit_date'].max().date()}")
print(f"  Months covered : {out['report_month'].nunique()}")
print(f"\nTop sub-types:")
print(out['sub_type'].value_counts().head(10).to_string())
print(f"\nTop work types:")
print(out['work_type'].value_counts().head(8).to_string())
print(f"\nSample rows:")
print(out[['permit_date','folder_number','sub_type','address','valuation','dwell_units']].head(6).to_string())
