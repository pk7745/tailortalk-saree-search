"""
Central configuration for the TailorTalk Saree Visual Search project.
Keep every tunable knob here so ingest.py and the app always agree.
"""
import os

# Prevent OpenMP multiple runtime conflict on Windows (PyTorch + FAISS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---- Paths -------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
INDEX_PATH = os.path.join(DATA_DIR, "saree_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.parquet")
FAILED_LOG = os.path.join(DATA_DIR, "failed_downloads.csv")

# ---- Embedding model -----------------------------------------------------
# open_clip pretrained checkpoint. ViT-B-32 is fast and light enough to
# comfortably re-embed 1000+ images and to run on Streamlit Cloud's free CPU
# tier at query time (no GPU available there).
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ---- Fusion weights --------------------------------------------------
# Final vector = normalize( concat( CLIP_WEIGHT * norm(clip_emb),
#                                    COLOR_WEIGHT * norm(color_hist) ) )
# CLIP captures overall garment semantics + coarse pattern/texture.
# The HSV color histogram captures precise colour-combination similarity,
# which plain CLIP tends to under-weight for a single-category catalogue
# like this one (every image IS a saree, so CLIP's dominant signal is
# "saree-ness", not "which saree"). Tuned by manual A/B on ~15 query images
# (see README > Search Quality).
CLIP_WEIGHT = 0.7
COLOR_WEIGHT = 0.3
COLOR_HIST_BINS = (8, 8, 8)  # H, S, V bins -> 512-dim histogram

# ---- Search ---------------------------------------------------------
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

# ---- Download behaviour (ingest.py) ----------------------------------
DOWNLOAD_TIMEOUT = 15
DOWNLOAD_RETRIES = 2
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TailorTalkIndexer/1.0)"
}

# ---- LLM ---------------------------------------------------------------
# Google Gemini free tier (no credit card needed): https://aistudio.google.com/apikey
GEMINI_MODEL = "gemini-2.0-flash"
