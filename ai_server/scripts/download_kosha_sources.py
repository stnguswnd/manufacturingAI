from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from rag_pipeline_utils import (
    HEADERS,
    KOSHA_API_RESPONSE_DIR,
    KOSHA_FILES_DIR,
    KOSHA_INDEX_JSONL_PATH,
    KOSHA_INDEX_JSON_PATH,
    PROJECT_ROOT,
    classify_kosha_doc,
    ensure_dirs,
    extension_from_headers_or_url,
    load_manifest,
    normalize_items,
    safe_filename,
    write_json,
    write_jsonl,
)


def env_value(name: str, default: str = '') -> str:
    for env_path in [PROJECT_ROOT / '.env', PROJECT_ROOT / 'ai_server' / '.env']:
        if env_path.exists():
            load_dotenv(env_path, override=False)
    return os.getenv(name, default).strip()


def parse_response_json(response: requests.Response) -> dict | None:
    try:
        data = response.json()
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def dedup_key(item: dict) -> str:
    return '|'.join(str(item.get(k) or '').strip() for k in ['techGdlnNo', 'techGdlnNm', 'fileDownlUrl'])


def download_file(url: str, *, title: str, doc_no: str, force: bool = False, timeout: int = 60) -> tuple[str | None, str | None]:
    if not url:
        return None, None
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        ext = extension_from_headers_or_url(dict(response.headers), url)
        base = safe_filename(f'{doc_no}_{title}' if doc_no else title)
        out = KOSHA_FILES_DIR / f'{base}{ext}'
        if out.exists() and not force:
            return str(out.relative_to(PROJECT_ROOT)), None
        out.write_bytes(response.content)
        return str(out.relative_to(PROJECT_ROOT)), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def download_kosha_sources(
    *,
    keywords: list[str],
    pages: int,
    num_rows: int,
    force: bool = False,
) -> dict:
    ensure_dirs()
    api_key = env_value('KOSHA_API_KEY')
    if not api_key:
        raise SystemExit('KOSHA_API_KEY is missing. Add it to .env or export it in the shell.')
    manifest = load_manifest()
    config = manifest.get('kosha_api') or {}
    endpoint = config.get('endpoint') or 'https://apis.data.go.kr/B552468/koshaguide/getKoshaGuide'
    call_api_id = env_value('KOSHA_CALL_API_ID', str(config.get('callApiId') or '1050'))
    unique: dict[str, dict] = {}
    api_failures: list[dict] = []
    file_failures: list[dict] = []

    for keyword in keywords:
        for page in range(1, pages + 1):
            params = {
                'serviceKey': api_key,
                'pageNo': page,
                'numOfRows': num_rows,
                'callApiId': call_api_id,
                'techGdlnNm': keyword,
            }
            response_path = KOSHA_API_RESPONSE_DIR / f'kosha_search_{safe_filename(keyword)}_page_{page}.json'
            try:
                response = requests.get(endpoint, params=params, headers=HEADERS, timeout=30)
                data = parse_response_json(response)
                if data is None:
                    raw_preview = response.text[:1000]
                    response_path.write_text(response.text, encoding='utf-8', errors='ignore')
                    api_failures.append({'keyword': keyword, 'page': page, 'url': response.url, 'status_code': response.status_code, 'response_preview': raw_preview})
                    time.sleep(1)
                    continue
                write_json(response_path, data)
                for item in normalize_items(data):
                    item = dict(item)
                    key = dedup_key(item)
                    if not key.strip('|') or key in unique:
                        continue
                    title = str(item.get('techGdlnNm') or '').strip()
                    metadata = classify_kosha_doc(title)
                    file_url = str(item.get('fileDownlUrl') or '').strip()
                    local_path = None
                    download_error = None
                    if file_url:
                        local_path, download_error = download_file(file_url, title=title, doc_no=str(item.get('techGdlnNo') or ''), force=force)
                        if download_error:
                            file_failures.append({'title': title, 'url': file_url, 'error': download_error})
                    unique[key] = {
                        'source': 'KOSHA',
                        'search_keyword': keyword,
                        'techGdlnNo': item.get('techGdlnNo'),
                        'techGdlnNm': title,
                        'techGdlnKnd': item.get('techGdlnKnd'),
                        'fileDownlUrl': file_url,
                        'local_path': local_path,
                        'download_error': download_error,
                        **metadata,
                    }
            except Exception as exc:
                api_failures.append({'keyword': keyword, 'page': page, 'url': endpoint, 'error': f'{type(exc).__name__}: {exc}'})
            time.sleep(1)

    rows = sorted(unique.values(), key=lambda row: (str(row.get('project_priority')), str(row.get('techGdlnNm'))))
    index = {'documents': rows, 'api_failures': api_failures, 'file_failures': file_failures}
    write_json(KOSHA_INDEX_JSON_PATH, index)
    write_jsonl(KOSHA_INDEX_JSONL_PATH, rows)
    return {'documents': rows, 'api_failures': api_failures, 'file_failures': file_failures}


def keyword_list(args) -> list[str]:
    manifest = load_manifest()
    api = manifest.get('kosha_api') or {}
    if args.keyword:
        return [args.keyword]
    keywords = list(api.get('keywords_primary') or [])
    if args.include_secondary:
        keywords.extend(api.get('keywords_secondary') or [])
    return list(dict.fromkeys(str(item).strip() for item in keywords if str(item).strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description='Download KOSHA guide API responses and linked files.')
    parser.add_argument('--keyword', help='Search one specific techGdlnNm keyword.')
    parser.add_argument('--include-secondary', action='store_true', help='Include secondary manifest keywords.')
    parser.add_argument('--pages', type=int, default=1, help='Pages per keyword.')
    parser.add_argument('--num-rows', type=int, default=20, help='Rows per page.')
    parser.add_argument('--force', action='store_true', help='Re-download files even if they exist.')
    args = parser.parse_args()
    result = download_kosha_sources(keywords=keyword_list(args), pages=args.pages, num_rows=args.num_rows, force=args.force)
    print(f'documents={len(result["documents"])} api_failures={len(result["api_failures"])} file_failures={len(result["file_failures"])}')
    if result['api_failures']:
        first = result['api_failures'][0]
        print('first_api_failure=', json.dumps(first, ensure_ascii=False)[:1200])


if __name__ == '__main__':
    main()
