from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_SERVER_DIR = PROJECT_ROOT / 'ai_server'
RAG_DATA_DIR = AI_SERVER_DIR / 'data'
MANIFEST_PATH = RAG_DATA_DIR / 'rag_source_manifest.yaml'
RAW_RAG_DIR = RAG_DATA_DIR / 'raw' / 'rag_sources'
PROCESSED_DIR = RAG_DATA_DIR / 'processed'
KOSHA_API_RESPONSE_DIR = RAW_RAG_DIR / 'kosha' / 'api_responses'
KOSHA_FILES_DIR = RAW_RAG_DIR / 'kosha' / 'files'
RAG_DOCUMENTS_PATH = PROCESSED_DIR / 'rag_documents.jsonl'
RAG_CHUNKS_PATH = PROCESSED_DIR / 'rag_chunks.jsonl'
KOSHA_INDEX_JSON_PATH = PROCESSED_DIR / 'kosha_download_index.json'
KOSHA_INDEX_JSONL_PATH = PROCESSED_DIR / 'kosha_download_index.jsonl'
RAG_CORPUS_REPORT_PATH = PROCESSED_DIR / 'rag_corpus_report.md'

HEADERS = {'User-Agent': 'ManufacturingAIResearchBot/0.1 educational project'}


def ensure_dirs() -> None:
    for path in [RAW_RAG_DIR, KOSHA_API_RESPONSE_DIR, KOSHA_FILES_DIR, PROCESSED_DIR, RAG_DATA_DIR / 'vector_db' / 'chroma']:
        path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')


def normalize_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload or not isinstance(payload, dict):
        return []
    body = payload.get('body')
    if not isinstance(body, dict):
        return []
    items = body.get('items')
    if not items:
        return []
    item = items.get('item') if isinstance(items, dict) else None
    if not item:
        return []
    if isinstance(item, list):
        return [x for x in item if isinstance(x, dict)]
    if isinstance(item, dict):
        return [item]
    return []


def safe_filename(value: str, *, max_length: int = 150) -> str:
    value = unicodedata.normalize('NFKC', value or '').strip()
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', '_', value)
    value = re.sub(r'\s+', '_', value)
    value = re.sub(r'_+', '_', value).strip('._ ')
    if not value:
        value = 'untitled'
    if len(value) > max_length:
        value = value[:max_length].rstrip('._ ')
    return value


def extension_from_headers_or_url(headers: dict[str, str], url: str) -> str:
    disposition = headers.get('Content-Disposition') or headers.get('content-disposition') or ''
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition)
    if match:
        suffix = Path(unquote(match.group(1))).suffix.lower()
        if suffix:
            return suffix
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix:
        return suffix
    content_type = (headers.get('Content-Type') or headers.get('content-type') or '').split(';')[0].strip().lower()
    return {
        'application/pdf': '.pdf',
        'application/zip': '.zip',
        'application/x-hwp': '.hwp',
        'application/haansofthwp': '.hwp',
        'application/vnd.hancom.hwpx': '.hwpx',
        'text/html': '.html',
    }.get(content_type, '.bin')


def classify_kosha_doc(title: str) -> dict[str, Any]:
    text = title or ''
    machine_terms = ['방호', '안전장치', '끼임', '회전', '비상정지', '인터록', '프레스', '공작기계', '컨베이어', '절삭']
    strong_machine_terms = ['방호', '안전장치', '끼임', '회전', '비상정지', '인터록', '프레스', '컨베이어', '절삭']
    maintenance_terms = ['정비', '점검', '보수', '유지관리', '작업 전']
    work_env_terms = ['작업환경측정', '소음', '분진', '보건', '유해물질', '화학', '환기']
    construction_terms = ['건설', '추락', '굴착']
    if any(term in text for term in strong_machine_terms):
        return {
            'doc_type': 'korean_machine_safety',
            'safety_gate': 'machine_guarding',
            'failure_modes': ['OSF', 'TWF'],
            'related_signals': ['rotational_speed_rpm', 'torque_nm', 'tool_wear_min'],
            'project_priority': 'high',
            'retrieval_scope': 'default',
            'use_case': '기계 방호/안전장치/회전부 관련 한국어 안전 근거',
        }
    if any(term in text for term in maintenance_terms):
        return {
            'doc_type': 'korean_maintenance_guidance',
            'safety_gate': 'maintenance_check',
            'failure_modes': ['OSF', 'TWF', 'PWF', 'HDF'],
            'related_signals': ['torque_nm', 'tool_wear_min', 'rotational_speed_rpm', 'air_temperature_k', 'process_temperature_k'],
            'project_priority': 'high',
            'retrieval_scope': 'default',
            'use_case': '정비/점검/유지관리 질문 및 보고서 보조',
        }
    if any(term in text for term in machine_terms):
        return {
            'doc_type': 'korean_machine_safety',
            'safety_gate': 'machine_guarding',
            'failure_modes': ['OSF', 'TWF'],
            'related_signals': ['rotational_speed_rpm', 'torque_nm', 'tool_wear_min'],
            'project_priority': 'high',
            'retrieval_scope': 'default',
            'use_case': '기계 방호/안전장치/회전부 관련 한국어 안전 근거',
        }
    if any(term in text for term in work_env_terms):
        return {
            'doc_type': 'korean_work_environment_guidance',
            'safety_gate': 'work_environment_check',
            'failure_modes': [],
            'related_signals': ['air_temperature_k', 'process_temperature_k'],
            'project_priority': 'low',
            'retrieval_scope': 'restricted',
            'use_case': '작업환경/보건 관련 질문 또는 보고서 보조',
        }
    if any(term in text for term in construction_terms):
        return {
            'doc_type': 'korean_general_safety_other',
            'safety_gate': 'general_safety',
            'failure_modes': [],
            'related_signals': [],
            'project_priority': 'low',
            'retrieval_scope': 'restricted',
            'use_case': '현재 AI4I 제조 예측정비 MVP와 직접 관련성은 낮음',
        }
    return {
        'doc_type': 'korean_safety_reference',
        'safety_gate': 'general_safety',
        'failure_modes': [],
        'related_signals': [],
        'project_priority': 'medium',
        'retrieval_scope': 'restricted',
        'use_case': '한국어 안전 참고자료',
    }


def clean_text(text: str) -> str:
    text = re.sub(r'\r\n?', '\n', text or '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding='utf-8', errors='ignore'), 'lxml')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'svg']):
        tag.decompose()
    return clean_text('\n'.join(line.strip() for line in soup.get_text('\n').splitlines() if line.strip()))


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return clean_text('\n'.join(page.extract_text() or '' for page in reader.pages))


def hwpx_to_text(path: Path) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.xml'):
                continue
            try:
                root = ElementTree.fromstring(zf.read(name))
            except ElementTree.ParseError:
                continue
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
    return clean_text('\n'.join(texts))


def extract_text(path: Path) -> tuple[str, str, str | None]:
    suffix = path.suffix.lower()
    try:
        if suffix in {'.html', '.htm'}:
            return html_to_text(path), 'success', None
        if suffix == '.pdf':
            return pdf_to_text(path), 'success', None
        if suffix == '.hwpx':
            return hwpx_to_text(path), 'success', None
        if suffix == '.txt':
            return clean_text(path.read_text(encoding='utf-8', errors='ignore')), 'success', None
        if suffix == '.hwp':
            return '', 'failed', 'binary .hwp extraction is not supported in the default pipeline'
        return '', 'failed', f'unsupported file extension: {suffix or "none"}'
    except Exception as exc:
        return '', 'failed', f'{type(exc).__name__}: {exc}'


def sentence_aware_chunks(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150, min_chunk_size: int = 120) -> list[str]:
    normalized = clean_text(text).replace('\n', ' ')
    if len(normalized) < min_chunk_size:
        return []
    sentences = re.split(r'(?<=[.!?。！？다요함음])\s+', normalized)
    chunks: list[str] = []
    current = ''
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f'{current} {sentence}'.strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if len(current) >= min_chunk_size:
            chunks.append(current)
        overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ''
        current = f'{overlap} {sentence}'.strip()
        while len(current) > chunk_size:
            part = current[:chunk_size]
            chunks.append(part)
            current = current[max(0, chunk_size - chunk_overlap):].strip()
    if len(current) >= min_chunk_size:
        chunks.append(current)
    return chunks


def counters(rows: list[dict[str, Any]], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            counter.update(str(item) for item in value)
        elif value is not None:
            counter[str(value)] += 1
    return counter
