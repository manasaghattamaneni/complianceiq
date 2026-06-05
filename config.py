# config.py
# Single source of truth for all app constants
# If you need to change any value — change it here only

# ---- App Info ----
APP_NAME = "ComplianceIQ"
APP_VERSION = "1.0.0"

# ---- File Upload Limits ----
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_TYPES = ["pdf", "docx", "txt", "csv"]

# ---- Chunking Settings ----
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
LARGE_DOC_THRESHOLD = 50000  # characters
LARGE_DOC_CHUNK_SIZE = 800
LARGE_DOC_OVERLAP = 100

# ---- ChromaDB Settings ----
COLLECTION_PREFIX = "complianceiq_"
MAX_RETRIEVAL_RESULTS = 3

# ---- AI Settings ----
AI_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_QUESTION_LENGTH = 500
RATE_LIMIT_SECONDS = 1

# ---- Logging ----
LOG_FILE = "complianceiq.log"
LOG_LEVEL = "INFO"

# ---- Security ----
DANGEROUS_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "disregard all previous",
    "you are now a",
    "pretend you are",
    "act as if you are",
    "jailbreak",
    "forget your instructions",
    "new instructions:",
]
