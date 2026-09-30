"""Publish an allowlist of measurements; never copy raw logs or prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from analyze_server_round import analyze

REQUEST_FIELDS={'request_id','doc_id','dataset_record_id','prompt_sha256','question_sha256',
                'context_prefix_sha256','expected_prompt_tokens','status','prompt_tokens',
                'cached_tokens','output_tokens','elapsed_ms','output_token_sha256'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    raw=Path(a.raw); out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    complete=raw/'runs/decision-screen-v2'
    count=0; trials=[]
    for folder in sorted(complete.iterdir()):
        if not folder.is_dir():
            continue
        summary=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        if summary['status']!='completed' or len(rows)!=128 or summary['success_count']!=128:
            raise ValueError('incomplete run: '+folder.name)
        if len({r['request_id'] for r in rows})!=128:
            raise ValueError('duplicate IDs')
        for r in rows:
            if set(r)-REQUEST_FIELDS or r['status']!='ok':
                raise ValueError('unexpected public field or failed request')
            if r['prompt_tokens']!=r['expected_prompt_tokens']:
                raise ValueError('tokenizer mismatch')
            if not 0<=r['cached_tokens']<=r['prompt_tokens']:
                raise ValueError('invalid cached token count')
            if r['output_tokens']!=summary['output_tokens_per_request']:
                raise ValueError('generation length mismatch')
        dest=out/folder.name;dest.mkdir()
        shutil.copyfile(folder/'summary.json',dest/'summary.json')
        (dest/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        count+=len(rows)
        trials.append({'name':folder.name,'status':'completed','requests':len(rows),
                       'manifest_sha256':summary['manifest_sha256']})
    partial=raw/'runs/decision-screen/0-hot_scan-stock/requests.jsonl'
    rows=[json.loads(s) for s in partial.read_text().splitlines()]
    if any(set(r)-REQUEST_FIELDS for r in rows):
        raise ValueError('unexpected partial measurement field')
    dest=out/'initial-timeout';dest.mkdir()
    (dest/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (dest/'summary.json').write_text(json.dumps({'status':'infrastructure_timeout',
        'success_count':len(rows),'request_count':128,'policy':'stock',
        'reason':'180-second process cap included first-time CUDA compilation; partial run not used in comparisons'},indent=2)+'\n')
    result=analyze(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    profile=json.loads((raw/'runtime-profile.json').read_text())
    profile['experiment_code_commit']='8e9bc89'
    profile['original_plan_note']='Core 32-token comparisons: one pair per scene plus original repeat; short-output comparisons supplemented to three pairs per pressure scene.'
    (out/'runtime-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (out/'trial-register.json').write_text(json.dumps({'completed_runs':len(trials),
        'completed_requests':count,'partial_requests':len(rows),'trials':trials},indent=2)+'\n')
    # Keep measurement byte hashes identical on Windows, Linux and Git.
    for path in out.rglob('*'):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    checksums={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'measurement-checksums.json').write_bytes((json.dumps(checksums,indent=2)+'\n').encode())
    print('Exported',len(trials),'completed runs;',count,'completed requests;',len(rows),'partial records')


if __name__=='__main__':
    main()
