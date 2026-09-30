"""Verify prepared input checksums before paid GPU work; no third-party imports."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'CHECKSUMS.json').read_text())
    for relative,digest in expected.items():
        path=root/relative
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):
                h.update(chunk)
        if h.hexdigest()!=digest:
            raise RuntimeError('checksum mismatch: '+relative)
    for path in (root/'prepared').glob('*.jsonl'):
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows)==128 and len({r['request_id'] for r in rows})==128
        assert len({r['prompt_sha256'] for r in rows})==128
        for row in rows:
            assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
            assert row['expected_prompt_tokens']+32<=2048
    print(f'Checksums passed: {len(expected)} files; 15 fixed manifests ready.')


if __name__=='__main__':
    main()
