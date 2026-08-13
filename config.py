"""
Central configuration for the TailorTalk Saree Visual Search project.
Keep every tunable knob here so ingest.py and the app always agree.
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# If not in env, check api_key.txt
if not os.environ.get("GOOGLE_API_KEY"):
    for key_file in [
        os.path.join(os.path.dirname(__file__), "api_key.txt"),
        os.path.join(os.path.dirname(__file__), "..", "api_key.txt"),
    ]:
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GOOGLE_API_KEY"] = val
                            break
                    elif line and not line.startswith("#") and len(line) > 20:
                        os.environ["GOOGLE_API_KEY"] = line
                        break

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

# ---- 3-Way Fusion weights (Whole Image + Color Histogram + Border/Pallu Crop) -
# Final vector = normalize( concat( CLIP_WEIGHT * norm(clip_whole),
#                                    COLOR_WEIGHT * norm(color_hist),
#                                    BORDER_WEIGHT * norm(clip_border) ) )
# 1. CLIP_WEIGHT: Captures overall garment silhouette & drapery (0.55).
# 2. COLOR_WEIGHT: Captures exact multi-color HSV palette distribution (0.25).
# 3. BORDER_WEIGHT: Captures fine-grained border & pallu weave motifs (0.20).
CLIP_WEIGHT = 0.55
COLOR_WEIGHT = 0.25
BORDER_WEIGHT = 0.20
COLOR_HIST_BINS = (8, 8, 8)  # H, S, V bins -> 512-dim histogram
BORDER_BOTTOM_RATIO = 0.35
BORDER_RIGHT_RATIO = 0.35

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
GEMINI_MODEL = "gemini-2.5-flash"
