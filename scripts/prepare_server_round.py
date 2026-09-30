"""Prepare private prompt manifests plus public hash-only workload audits."""
import argparse
import json
from pathlib import Path
from server_workloads import build_workload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--audit',required=True)
    p.add_argument('--count',type=int,default=128)
    a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    audit=Path(a.audit); audit.mkdir(parents=True,exist_ok=False)
    for rep in range(5):
        for scene in ['hot_scan','mixed_zipf','roomy_control']:
            rows,metadata=build_workload(a.source,tokenizer,scene,seed=20260930+rep,count=a.count)
            if any(r['expected_prompt_tokens']+32>2048 for r in rows):
                raise ValueError('prompt exceeds fixed model context budget')
            name=f'{scene}-{rep}'
            (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (audit/(name+'.json')).write_text(json.dumps({'metadata':metadata,
                'requests':[{k:v for k,v in r.items() if k!='prompt'} for r in rows]},indent=2)+'\n')
            print(name,metadata['complete_row_count'],metadata['source_sha256'],flush=True)


if __name__=='__main__':
    main()
