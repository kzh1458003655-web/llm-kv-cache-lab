"""Retokenize the locked source for 9B; write private inputs and public audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from nineb_artifacts import sha256, verify_model
from server_workloads import build_workload

PROGRESS_FILE = None


def main():
    global PROGRESS_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'manifest', 'protocol', 'output', 'audit'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if sha256(args.source) != protocol['source']['sha256']:
        raise ValueError('source differs from the complete locked LooGLE file')
    verification = verify_model(args.manifest, args.model, metadata_only=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    args.audit.mkdir(parents=True, exist_ok=False)
    PROGRESS_FILE = args.output / 'preparation-state.json'
    PROGRESS_FILE.write_text(json.dumps({'status': 'incomplete', 'note': 'partial inputs are not a valid upload kit; preserve failures and rerun to a new directory'}) + '\n')
    index = {'model_revision': verification['revision'],
             'protocol_sha256': sha256(args.protocol), 'files': []}
    count = protocol['online_workload']['requests_per_measured_run']
    for target in protocol['online_workload']['input_token_targets']:
        for scene in ('hot_scan', 'mixed_zipf'):
            for rep in range(5):
                seed = 20261001 + rep
                rows, metadata = build_workload(args.source, tokenizer, scene,
                    seed=seed, count=count, context_tokens=target)
                if metadata['partial_line_count'] or metadata['source_sha256'] != protocol['source']['sha256']:
                    raise ValueError('incomplete or changed source')
                rng = random.Random(seed + 710000)
                elapsed = 0.0
                for row in rows:
                    elapsed += rng.expovariate(1.0)
                    row['arrival_units'] = elapsed
                    row['prompt_token_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
                    row['input_token_ids_sha256'] = hashlib.sha256(
                        json.dumps(row['prompt_token_ids'], separators=(',', ':')).encode()).hexdigest()
                    if len(row['prompt_token_ids']) != row['expected_prompt_tokens']:
                        raise ValueError('tokenization mismatch')
                    if row['expected_prompt_tokens'] + protocol['online_workload']['output_tokens'] > protocol['runtime']['max_model_len']:
                        raise ValueError('request exceeds fixed model context')
                name = f'{scene}-ctx{target}-rep{rep}'
                private = args.output / (name + '.jsonl')
                private.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                                          for row in rows), encoding='utf-8', newline='\n')
                public_rows = [{key: value for key, value in row.items()
                                if key not in ('prompt', 'prompt_token_ids')} for row in rows]
                audit = {'metadata': metadata, 'model_revision': verification['revision'],
                         'private_file_sha256': sha256(private),
                         'arrival_units_definition': 'cumulative Exp(1) draws; seconds = units / frozen request_rate',
                         'requests': public_rows}
                (args.audit / (name + '.json')).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                index['files'].append({'name': name, 'count': len(rows), 'sha256': sha256(private),
                                       'scene': scene, 'context_tokens': target, 'rep': rep,
                                       'min_input_tokens': min(row['expected_prompt_tokens'] for row in rows),
                                       'max_input_tokens': max(row['expected_prompt_tokens'] for row in rows)})
                print(name, len(rows), flush=True)
    (args.audit / 'index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    PROGRESS_FILE.write_text(json.dumps({'status': 'complete', 'files': len(index['files'])}) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if PROGRESS_FILE is not None:
            PROGRESS_FILE.write_text(json.dumps({'status': 'failed', 'error_type': type(exc).__name__,
                'note': 'failure artifacts preserved; choose new output/audit directories for a retry'}) + '\n')
        raise
