# Agent Brief — Build, Test, and Deploy TailorTalk Saree Search

You are picking up a scaffolded, syntax-checked, but **not-yet-fully-run**
project. The code has been validated with synthetic data (mock FAISS
vectors, mock color histograms). It has NOT yet been run against the real
1,074-row image catalogue, because that requires downloading images from
`byrappasilk.in`, which needs real network access. Your job is to make it
100% real, verified, and deployed.

Do these steps **in order**. Do not skip step 2 (full ingest) or step 3
(selftest) — a partially-indexed catalogue or an unverified pipeline is
not an acceptable final state.

## Step 1 — Environment
```bash
cd tailortalk-saree-search
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Confirm `torch`, `open_clip`, `faiss`, `langchain`, `streamlit` import
without error before proceeding.

## Step 2 — Full ingest (ALL 1,074 rows, not a sample)
```bash
python ingest.py
```
- This must process **every row** in `data/products.csv` — do not stop at
  a `--limit`. If you use `--limit` first to smoke-test, that's fine, but
  the final run before deployment must be the full, unlimited run.
- It is resumable: if it's interrupted (network hiccup, timeout), just
  re-run `python ingest.py` again — it skips URLs already indexed.
- After it finishes, check `data/failed_downloads.csv` if it exists.
  Coverage should be **at least 90% of the 1,074 rows** successfully
  indexed (some product image URLs may be dead/404 — that's expected and
  fine, but a large failure rate means something's wrong with network
  access or the download logic, and needs investigating before moving on).
- Confirm `data/saree_index.faiss` and `data/metadata.parquet` now exist
  and are non-trivial in size (a few MB, not a few KB).

## Step 3 — Run the self-test suite
```bash
export GOOGLE_API_KEY="<your free Gemini key from https://aistudio.google.com/apikey>"
python selftest.py
```
This must **exit 0 with "All checks passed."** before you move on. It
checks, using the real indexed data:
1. Coverage — most of the 1,074 catalogue rows made it into the index.
2. Identity — re-querying a real catalogue image returns itself as the #1
   match with a near-1.0 score (proves the embedding pipeline isn't
   broken/shuffled).
3. Discrimination — similarity scores actually spread out across the
   catalogue (catches a degenerate embedding where everything scores the
   same, which would explain "loose, generic results").
4. Schema — every result dict has exactly the documented fields.
5. Agent intent — the LLM correctly calls `find_similar_sarees` when asked
   for similar items, and does **not** call it for unrelated chit-chat
   ("hi, how are you?"). This is the "acts exactly according to what the
   human is asking" requirement — verify it end to end, not just by
   reading the prompt.

**If anything fails, fix the root cause and re-run selftest.py until it's
fully green.** Do not deploy on a partial pass. Likely failure modes and
what they mean:
- Identity check fails → embedding function or FAISS indexing has a bug
  (e.g. vector order mismatch between `metadata.parquet` rows and FAISS
  index rows) — this is a correctness bug, fix it, don't just retune weights.
- Discrimination check fails (spread too small) → the embedding really is
  too generic. Try raising `COLOR_WEIGHT` in `config.py` (e.g. 0.3 → 0.4)
  and re-run `ingest.py --rebuild` + `selftest.py`.
- Agent intent check fails → adjust `SYSTEM_PROMPT` in `agent.py` to be
  more explicit about when to call vs. not call the tool, re-test.

## Step 4 — Manual spot-check (do this yourself, don't skip)
```bash
streamlit run app.py
```
Pull up the app and, using **at least 6 different query images spanning
different fabric types** in the catalogue (e.g. a Banarasi, an Organza, an
Ajrakh print, a Pashmina, a Linen, a plain Satin), for each one:
- Upload it and ask "find similar sarees to this."
- Confirm the tool is called (image grid renders) and the top matches are
  visually plausible — similar color family and fabric/print, not random
  sarees that merely share "generic saree" shape.
- Also send an unrelated message with no image uploaded (e.g. "what's the
  weather like") and confirm the agent does NOT try to search and instead
  responds conversationally / asks for an image if relevant.
- Try a follow-up in the same chat ("show me 8 instead") and confirm
  `top_k` is respected and conversation context carries over.

Note down any query where results look poor — if there's a systematic
pattern (e.g. always confuses two similar-looking fabric categories),
that's useful to record in the README's limitations section, since the
brief explicitly asks for "any assumptions or trade-offs."

## Step 5 — Deploy
1. Push the full repo to GitHub, **including** `data/saree_index.faiss`
   and `data/metadata.parquet` (small, derived files — must be committed,
   they are NOT in `.gitignore`). Do not commit `.streamlit/secrets.toml`
   (it is gitignored — only commit `secrets.toml.example`).
2. Deploy on Streamlit Community Cloud (share.streamlit.io) pointing at
   `app.py`, with `GOOGLE_API_KEY` set in the app's Secrets panel.
3. Once live, re-run the Step 4 manual spot-check against the **deployed**
   URL, not just localhost — hosting environment differences (CPU
   throttling, cold starts, missing files) are a common last-mile failure.
4. Fill in the live app URL and GitHub repo URL at the top of `README.md`
   and commit that update.

## Definition of done
- [ ] `python ingest.py` completed on all 1,074 rows, ≥90% coverage.
- [ ] `python selftest.py` exits 0, all 5+ checks pass.
- [ ] Manual spot-check across ≥6 fabric types shows visually coherent
      top-5 matches (same color family + fabric/print as query).
- [ ] Agent correctly distinguishes "search" requests from general chat.
- [ ] App deployed and re-verified live, not just localhost.
- [ ] README's live URL / repo URL fields filled in, any real limitations
      found during testing added to its "Assumptions & trade-offs" section.
