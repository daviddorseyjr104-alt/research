import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# On Railway: set PERSISTENT_DIR=/data and mount a volume there
# Locally: defaults to project root (same behaviour as before)
_persist = Path(os.environ.get("PERSISTENT_DIR", str(BASE_DIR)))
OUTPUTS_DIR = _persist / "outputs"
DB_PATH = _persist / "apw_knowledge.db"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("APW_MODEL", "claude-opus-4-8")
DEEP_MODEL = os.environ.get("APW_DEEP_MODEL", "claude-opus-4-8")

ORG_NAME = "Africa Pension Watch"
ORG_DESCRIPTION = (
    "An independent advocacy group and think tank promoting best practices in "
    "pension management, governance, regulation, and investment across Africa."
)

REQUEST_DELAY = 1        # seconds between web requests
REQUEST_TIMEOUT = 15     # seconds per request
MAX_CONTENT_LENGTH = 50_000   # characters to store per document
MAX_PDF_PAGES = 50       # max pages to extract from PDFs
USER_AGENT = (
    "Mozilla/5.0 (compatible; AfricaPensionWatch-Research/1.0; "
    "+https://africapensionwatch.org)"
)
