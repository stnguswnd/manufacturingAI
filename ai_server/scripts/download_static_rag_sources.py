from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from rag_pipeline_utils import HEADERS, PROJECT_ROOT, ensure_dirs, load_manifest


def download_static_sources(*, force: bool = False, timeout: int = 30) -> dict:
    ensure_dirs()
    manifest = load_manifest()
    downloaded: list[str] = []
    skipped: list[str] = []
    failed_downloads: list[dict] = []
    for source in manifest.get('sources') or []:
        url = source.get('url')
        save_path = PROJECT_ROOT / source['save_path']
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not force:
            skipped.append(source['id'])
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            save_path.write_bytes(response.content)
            downloaded.append(source['id'])
        except Exception as exc:
            failed_downloads.append({'id': source.get('id'), 'url': url, 'error': f'{type(exc).__name__}: {exc}'})
        time.sleep(1)
    return {'downloaded': downloaded, 'skipped': skipped, 'failed_downloads': failed_downloads}


def main() -> None:
    parser = argparse.ArgumentParser(description='Download static OSHA/Haas RAG source HTML files from the manifest.')
    parser.add_argument('--force', action='store_true', help='Re-download files even if they already exist.')
    args = parser.parse_args()
    summary = download_static_sources(force=args.force)
    print(f'downloaded={len(summary["downloaded"])} skipped={len(summary["skipped"])} failed={len(summary["failed_downloads"])}')
    for failed in summary['failed_downloads']:
        print(f'failed id={failed["id"]} url={failed["url"]} error={failed["error"]}')


if __name__ == '__main__':
    main()
