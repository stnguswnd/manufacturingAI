from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv

AI_SERVER_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = AI_SERVER_DIR.parent

# Load both project-level and ai_server-level env files.
# Values already exported in the shell take precedence.
for _env in [PROJECT_ROOT / '.env', AI_SERVER_DIR / '.env']:
    if _env.exists():
        load_dotenv(_env, override=False)

DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
STORAGE_DIR = Path(os.getenv('STORAGE_DIR', AI_SERVER_DIR / 'storage'))
MODEL_DIR = STORAGE_DIR / 'models'
VECTOR_DIR = STORAGE_DIR / 'vector_store'
HISTORY_DIR = STORAGE_DIR / 'history'

AI4I_CSV = DATA_DIR / 'ai4i' / 'ai4i2020.csv'
MODEL_BUNDLE = MODEL_DIR / 'ai4i_model_bundle.joblib'
MODEL_METRICS = MODEL_DIR / 'ai4i_metrics.json'
CHUNKS_PATH = VECTOR_DIR / 'chunks.jsonl'
INDEX_PATH = VECTOR_DIR / 'tfidf_index.joblib'
HISTORY_DB_PATH = Path(os.getenv('HISTORY_DB_PATH', HISTORY_DIR / 'agent_runs.sqlite3'))
LANGGRAPH_CHECKPOINT_DB = Path(os.getenv('LANGGRAPH_CHECKPOINT_DB', STORAGE_DIR / 'checkpoints' / 'langgraph_checkpoints.sqlite3'))

APP_NAME = 'Manufacturing Domain AI Agent'
APP_VERSION = '0.3.0'
APP_ENV = os.getenv('APP_ENV', 'local').strip().lower()

# LLM configuration. This app is LLM-first: local/template execution is not a
# supported runtime mode.
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5.4-mini').strip()
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.2'))
LLM_TIMEOUT_SECONDS = float(os.getenv('LLM_TIMEOUT_SECONDS', '60'))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '1800'))
LLM_ENABLE_STRUCTURED_OUTPUT = os.getenv('LLM_ENABLE_STRUCTURED_OUTPUT', 'true').strip().lower() in {'1','true','yes','y'}
LLM_ALLOW_EXPENSIVE_MODELS = os.getenv('LLM_ALLOW_EXPENSIVE_MODELS', 'false').strip().lower() in {'1','true','yes','y'}
USD_KRW_EXCHANGE_RATE = float(os.getenv('USD_KRW_EXCHANGE_RATE', '1400'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '').strip() or None
OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID', '').strip() or None
OPENAI_PROJECT_ID = os.getenv('OPENAI_PROJECT_ID', '').strip() or None

# KOSHA public data API. Keys must be provided only through .env or the shell.
KOSHA_API_KEY = os.getenv('KOSHA_API_KEY', '').strip()
KOSHA_CALL_API_ID = os.getenv('KOSHA_CALL_API_ID', '1050').strip()

# Agent runtime controls.
AGENT_SUPERVISOR_LLM_REFINEMENT = os.getenv('AGENT_SUPERVISOR_LLM_REFINEMENT', 'true').strip().lower() in {'1','true','yes','y'}
AGENT_MAX_RAG_TOP_K = int(os.getenv('AGENT_MAX_RAG_TOP_K', '8'))
AGENT_MAX_REPLAN_ATTEMPTS = int(os.getenv('AGENT_MAX_REPLAN_ATTEMPTS', '2'))
RAG_MIN_NORMALIZED_SCORE = float(os.getenv('RAG_MIN_NORMALIZED_SCORE', '0.05'))
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', '2000'))
MAX_RECENT_RUNS = int(os.getenv('MAX_RECENT_RUNS', '3'))
MAX_SIMILAR_RUNS = int(os.getenv('MAX_SIMILAR_RUNS', '3'))
MAX_LONG_TERM_MEMORIES = int(os.getenv('MAX_LONG_TERM_MEMORIES', '5'))

# API protection. Disabled by default for local demos; enable in non-local deployments.
API_AUTH_ENABLED = os.getenv('API_AUTH_ENABLED', 'false').strip().lower() in {'1','true','yes','y'}
API_KEY = os.getenv('API_KEY', '').strip()
API_KEY_HEADER_NAME = os.getenv('API_KEY_HEADER_NAME', 'X-API-Key').strip() or 'X-API-Key'

# Comma-separated CORS allowlist. Defaults cover Streamlit and common local frontends.
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        'CORS_ALLOW_ORIGINS',
        'http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]

# Manufacturing domain configuration files.
DOMAIN_DIR = Path(os.getenv('DOMAIN_DIR', AI_SERVER_DIR / 'domain'))
APP_CONFIG_DIR = Path(__file__).resolve().parent
FAILURE_MODE_RETRIEVAL_POLICY = APP_CONFIG_DIR / 'failure_mode_retrieval_policy.yaml'

# Prices are USD per 1M tokens. Keep this list intentionally small and explicit
# so expensive models cannot be selected accidentally from the UI/API.
LLM_MODEL_CATALOG = {
    'gpt-5.4-mini': {
        'label': 'GPT-5.4 mini',
        'tier': 'standard',
        'input_per_1m': 0.75,
        'cached_input_per_1m': 0.075,
        'output_per_1m': 4.50,
        'selectable': True,
        'recommended': True,
    },
    'gpt-5.4': {
        'label': 'GPT-5.4',
        'tier': 'standard',
        'input_per_1m': 2.50,
        'cached_input_per_1m': 0.25,
        'output_per_1m': 15.00,
        'selectable': True,
        'recommended': False,
    },
    'gpt-5.5': {
        'label': 'GPT-5.5',
        'tier': 'expensive',
        'input_per_1m': 5.00,
        'cached_input_per_1m': 0.50,
        'output_per_1m': 30.00,
        'selectable': LLM_ALLOW_EXPENSIVE_MODELS,
        'recommended': False,
    },
}
LLM_SELECTABLE_MODELS = [model for model, info in LLM_MODEL_CATALOG.items() if info.get('selectable')]
