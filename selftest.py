"""
Automated self-test suite. Run this AFTER `python ingest.py` (full run, no
--limit) and BEFORE declaring the project done / deploying.

    python selftest.py

Exits non-zero and prints a clear failure reason if anything is wrong, so a
coding agent (or CI) can gate on it. Checks, in order:

  1. Coverage    - the FAISS index covers (close to) the whole catalogue.
  2. Identity     - embedding a catalogue image and searching returns
                    itself as the #1 match with near-1.0 score. This is
                    the single most important sanity check: if this fails,
                    the embedding/index pipeline itself is broken, no
                    amount of prompt tuning will fix bad results.
  3. Discrimination - two sarees of visibly different colour/fabric should
                    NOT be near-duplicates in the index (catches a
                    degenerate "everything looks the same" embedding).
  4. Schema       - search_similar_sarees() returns exactly the documented
                    fields, scores are sane floats.
  5. Metadata Filtering - filtered searches (e.g. max_price=3000) strictly
                    enforce the constraint across all returned results.
  6. Agent intent - (only if GOOGLE_API_KEY is set) the LLM correctly
                    calls the tool when asked for similar items, and does
                    NOT call it for an unrelated chit-chat message.
"""
from __future__ import annotations

import os
import random
import re
import sys

import pandas as pd

import config

FAIL = []
PASS = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"[PASS] {name}")
    else:
        FAIL.append((name, detail))
        print(f"[FAIL] {name}  -- {detail}")


def main():
    random.seed(7)

    # ---- 0. Files exist -------------------------------------------------
    if not (os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH)):
        print(
            "No index/metadata found. Run `python ingest.py` (full run, "
            "no --limit) before selftest.py."
        )
        sys.exit(1)

    import faiss
    from search_tool import search_similar_sarees, load_image_from_url, index_size

    index, meta = faiss.read_index(config.INDEX_PATH), pd.read_parquet(config.METADATA_PATH)
    total_rows = len(pd.read_csv(config.PRODUCTS_CSV).dropna(subset=["image_url"]))

    # ---- 1. Coverage ------------------------------------------------------
    coverage = index.ntotal / total_rows
    check(
        "Coverage >= 90% of catalogue indexed",
        coverage >= 0.90,
        f"indexed {index.ntotal}/{total_rows} ({coverage:.1%}). "
        f"Check data/failed_downloads.csv for skipped rows.",
    )

    # ---- 2 & 3. Identity + Discrimination (sample real catalogue images) --
    sample_idxs = random.sample(range(len(meta)), min(8, len(meta)))
    identity_ok = True
    identity_scores = []
    for i in sample_idxs:
        row = meta.iloc[i]
        try:
            img = load_image_from_url(row["image_url"])
        except Exception as e:  # noqa: BLE001
            print(f"  (skipping identity check for row {i}, download failed: {e})")
            continue
        results = search_similar_sarees(img, top_k=3)
        top = results[0] if results else None
        identity_scores.append(top["score"] if top else -1)
        if not top or top["image_url"] != row["image_url"] or top["score"] < 0.98:
            identity_ok = False
            print(f"  identity mismatch for {row['name']!r}: top match was "
                  f"{top['name'] if top else None!r} at score {top['score'] if top else None}")

    check(
        "Identity: querying a catalogue image returns itself as #1 (score>=0.98)",
        identity_ok,
        f"scores seen: {identity_scores}",
    )

    if len(sample_idxs) >= 2:
        a, b = meta.iloc[sample_idxs[0]], meta.iloc[sample_idxs[1]]
        try:
            img_a = load_image_from_url(a["image_url"])
            results = search_similar_sarees(img_a, top_k=len(meta))
            score_map = {r["image_url"]: r["score"] for r in results}
            spread = max(score_map.values()) - min(score_map.values())
            check(
                "Discrimination: similarity scores have real spread (not all near-identical)",
                spread > 0.15,
                f"score spread across catalogue = {spread:.3f} (want > 0.15)",
            )
        except Exception as e:  # noqa: BLE001
            check("Discrimination check ran", False, str(e))

    # ---- 4. Schema --------------------------------------------------------
    try:
        img = load_image_from_url(meta.iloc[sample_idxs[0]]["image_url"])
        results = search_similar_sarees(img, top_k=5)
        expected_keys = {"name", "sku", "price", "image_url", "product_link", "score"}
        schema_ok = all(expected_keys.issubset(r.keys()) for r in results) and len(results) <= 5
        check("Schema: results have documented fields, top_k respected", schema_ok, str(results[:1]))
    except Exception as e:  # noqa: BLE001
        check("Schema check ran", False, str(e))

    # ---- 5. Metadata Filter Enforcement -----------------------------------
    try:
        img = load_image_from_url(meta.iloc[sample_idxs[0]]["image_url"])
        filtered_results = search_similar_sarees(img, top_k=5, max_price=3000)
        filter_ok = len(filtered_results) > 0 and all(
            float(re.sub(r"[^\d.]", "", str(r["price"]))) <= 3000 for r in filtered_results
        )
        check(
            "Metadata Filtering: filtered query (max_price=3000) strictly enforces price <= 3000",
            filter_ok,
            f"Returned {len(filtered_results)} items, prices: {[r['price'] for r in filtered_results]}",
        )
    except Exception as e:  # noqa: BLE001
        check("Metadata Filtering check ran", False, str(e))

    # ---- 6. Agent intent (needs API key; skipped otherwise) --------------
    if os.environ.get("GOOGLE_API_KEY"):
        from agent import build_agent_executor

        img = load_image_from_url(meta.iloc[sample_idxs[0]]["image_url"])
        called = {"count": 0}

        def _on_results(r):
            called["count"] += 1

        try:
            import time
            time.sleep(2)
            executor = build_agent_executor(img, _on_results)
            
            # Retry up to 3 times on free-tier rate limit (429)
            for attempt in range(3):
                try:
                    executor.invoke({"input": "find me similar sarees to this one", "chat_history": []})
                    break
                except Exception as ex:
                    if "429" in str(ex) and attempt < 2:
                        time.sleep(25)
                    else:
                        raise
            check("Agent calls the tool when asked for similar items", called["count"] >= 1)

            called["count"] = 0
            time.sleep(2)
            executor2 = build_agent_executor(img, _on_results)
            for attempt in range(3):
                try:
                    executor2.invoke({"input": "hi, how are you?", "chat_history": []})
                    break
                except Exception as ex:
                    if "429" in str(ex) and attempt < 2:
                        time.sleep(25)
                    else:
                        raise
            check("Agent does NOT call the tool for unrelated chit-chat", called["count"] == 0)
        except Exception as e:  # noqa: BLE001
            check("Agent intent checks ran", False, str(e))
    else:
        print("[SKIP] Agent intent checks (GOOGLE_API_KEY not set in this shell)")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("All checks passed. Safe to deploy.")


if __name__ == "__main__":
    main()
