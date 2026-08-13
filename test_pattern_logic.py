"""
Mock-based regression suite for border/pallu/pattern feature.
Tests all logic WITHOUT loading FAISS index or CLIP model (avoids OpenMP DLL conflict on Windows).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pandas as pd
import numpy as np
import re

# ── bring in the pure-Python logic we want to test ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from search_tool import parse_query_intent, _matches_pattern, PATTERNS, COLORS, FABRICS

# ── build a tiny synthetic metadata frame ────────────────────────────────────
MOCK_ROWS = [
    {"name": "Golden Zari Border Banarasi Silk Saree",   "pattern": "golden zari", "material": "silk",   "work_type": "zari border", "price": "2500"},
    {"name": "Temple Border Kanchipuram Silk Saree",      "pattern": "temple border","material": "silk",  "work_type": "temple",      "price": "4800"},
    {"name": "Kadiyal Border Tussar Silk Saree",          "pattern": "kadiyal",     "material": "tussar", "work_type": "kadiyal border","price": "3200"},
    {"name": "Contrast Border Organza Saree",             "pattern": "contrast border","material":"organza","work_type": "contrast",   "price": "2200"},
    {"name": "Parrot Pallu Banarasi Saree",               "pattern": "parrot pallu","material": "silk",   "work_type": "pallu",       "price": "5500"},
    {"name": "Floral Embroidery Georgette Saree",         "pattern": "floral",      "material": "georgette","work_type":"embroidery",  "price": "1800"},
    {"name": "Plain Pink Chiffon Saree",                  "pattern": "",            "material": "chiffon","work_type": "",             "price": "1200"},
    {"name": "Silver Zari Border Mysore Silk Saree",      "pattern": "silver zari", "material": "silk",   "work_type": "zari border", "price": "6200"},
    {"name": "Applique Work Linen Saree",                 "pattern": "applique",    "material": "linen",  "work_type": "aplic work",  "price": "2900"},
    {"name": "Geometric Zari Banarasi Saree",             "pattern": "geometric zari","material":"silk",  "work_type": "geometric",   "price": "3700"},
]
meta = pd.DataFrame(MOCK_ROWS)

PASS = 0
FAIL = 0

def check(test_name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"[PASS] {test_name}")
        PASS += 1
    else:
        print(f"[FAIL] {test_name} — {detail}")
        FAIL += 1


print("="*80)
print("MOCK REGRESSION SUITE — BORDER/PALLU/PATTERN FEATURE")
print("="*80)

# ── Test 1: parse_query_intent extracts zari border ─────────────────────────
r1 = parse_query_intent("Show sarees with zari border")
check("1. parse_query_intent extracts 'zari border'", r1["pattern"] == "zari border",
      f"got: {r1['pattern']!r}")

# ── Test 2: parse_query_intent extracts temple border ───────────────────────
r2 = parse_query_intent("Show sarees with temple border under 5000")
check("2. parse_query_intent extracts 'temple border'", r2["pattern"] == "temple border",
      f"got: {r2['pattern']!r}")
check("2b. parse_query_intent extracts max_price 5000", r2["max_price"] == 5000,
      f"got: {r2['max_price']!r}")

# ── Test 3: parse_query_intent extracts kadiyal border ──────────────────────
r3 = parse_query_intent("Find kadiyal border sarees under 4000")
check("3. parse_query_intent extracts 'kadiyal border'", r3["pattern"] == "kadiyal border",
      f"got: {r3['pattern']!r}")

# ── Test 4: parse_query_intent extracts pallu ───────────────────────────────
r4 = parse_query_intent("Find sarees with parrot pallu")
check("4. parse_query_intent extracts 'parrot pallu'", r4["pattern"] == "parrot pallu",
      f"got: {r4['pattern']!r}")

# ── Test 5: _matches_pattern — exact match in name ──────────────────────────
row = meta[meta["name"].str.contains("Golden Zari")].iloc[0]
check("5. _matches_pattern 'golden zari' hits golden zari row", _matches_pattern("golden zari", row))

# ── Test 6: _matches_pattern — synonym expansion (zari border → golden zari) ─
check("6. _matches_pattern 'zari border' hits golden zari row via synonym",
      _matches_pattern("zari border", row))

# ── Test 7: _matches_pattern — temple border ────────────────────────────────
row7 = meta[meta["name"].str.contains("Temple")].iloc[0]
check("7. _matches_pattern 'temple border' hits temple border row", _matches_pattern("temple border", row7))

# ── Test 8: _matches_pattern — kadiyal border ───────────────────────────────
row8 = meta[meta["name"].str.contains("Kadiyal")].iloc[0]
check("8. _matches_pattern 'kadiyal border' hits kadiyal row", _matches_pattern("kadiyal border", row8))
check("8b. _matches_pattern 'kadiyal' hits kadiyal row", _matches_pattern("kadiyal", row8))

# ── Test 9: _matches_pattern — pallu ────────────────────────────────────────
row9 = meta[meta["name"].str.contains("Pallu")].iloc[0]
check("9. _matches_pattern 'parrot pallu' hits parrot pallu row", _matches_pattern("parrot pallu", row9))
check("9b. _matches_pattern 'pallu' hits parrot pallu row", _matches_pattern("pallu", row9))

# ── Test 10: _matches_pattern — None/empty returns True (no filtering) ──────
row10 = meta.iloc[0]
check("10. _matches_pattern None returns True (pass-through)", _matches_pattern(None, row10))
check("10b. _matches_pattern '' returns True (pass-through)", _matches_pattern("", row10))

# ── Test 11: _matches_pattern — plain row doesn't match temple border ───────
plain_row = meta[meta["name"].str.contains("Plain")].iloc[0]
check("11. _matches_pattern 'temple border' does NOT match plain row",
      not _matches_pattern("temple border", plain_row))

# ── Test 12: _matches_pattern — applique synonym ────────────────────────────
row12 = meta[meta["name"].str.contains("Applique")].iloc[0]
check("12. _matches_pattern 'applique work' hits applique row", _matches_pattern("applique work", row12))

# ── Test 13: No regression — parse_query_intent returns color correctly ──────
r13 = parse_query_intent("Show pink banarasi sarees under 3000")
check("13. parse_query_intent extracts color 'pink'", r13["color"] == "pink",
      f"got: {r13['color']!r}")
check("13b. parse_query_intent extracts fabric 'banarasi'", r13["fabric"] == "banarasi",
      f"got: {r13['fabric']!r}")
check("13c. parse_query_intent extracts max_price 3000", r13["max_price"] == 3000,
      f"got: {r13['max_price']!r}")

# ── Test 14: parse_query_intent with no pattern returns None ─────────────────
r14 = parse_query_intent("Show me some beautiful sarees")
check("14. parse_query_intent returns None pattern for generic query",
      r14["pattern"] is None, f"got: {r14['pattern']!r}")

# ── Test 15: PATTERNS vocabulary contains required terms ────────────────────
required = ["zari border", "golden zari", "temple border", "kadiyal border", "contrast border",
            "parrot pallu", "pallu", "border", "floral work", "applique work", "embroidery"]
missing = [p for p in required if p not in PATTERNS]
check("15. PATTERNS vocabulary contains all required terms",
      len(missing) == 0, f"Missing: {missing}")

print("="*80)
print(f"RESULTS: {PASS} PASSED, {FAIL} FAILED out of {PASS+FAIL} checks")
print("="*80)
if FAIL == 0:
    print("ALL CHECKS PASSED -- Border/pallu/pattern feature verified!")
else:
    print("SOME CHECKS FAILED -- Review above failures")
    sys.exit(1)
