"""Describe each independent run and matched pairs without pooled significance."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root):
    runs = {}
    for path in sorted(root.glob('*/summary.json')):
        summary = json.loads(path.read_text())
        data = path.parent/'requests.jsonl'
        rows = [json.loads(s) for s in data.read_text().splitlines()] if data.exists() else []
        durations = sorted(r['elapsed_ms'] for r in rows)
        total = sum(r['prompt_tokens'] for r in rows)
        cached = [r.get('cached_tokens') for r in rows]
        status=summary['status']
        if status=='completed' and (not rows or sum(durations)<=0
                or summary.get('success_count',len(rows))!=len(rows)
                or summary.get('request_count',len(rows))!=len(rows)
                or any('output_token_sha256' not in r for r in rows)):
            status='invalid_completed_artifact'
        runs[path.parent.name] = {'status':status, 'n':len(rows),
            'cached_tokens':sum(cached) if all(isinstance(x,int) for x in cached) else None,
            'input_tokens':total, 'p50_generate_ms':statistics.median(durations) if durations else None,
            'p95_generate_ms':durations[min(len(durations)-1,int(.95*(len(durations)-1)))] if durations else None,
            'sum_generate_ms':sum(durations), 'rows':rows}
    pairs=[]
    original_repeats=[]
    for name, run in runs.items():
        if name.endswith('-stock-repeat'):
            original=runs.get(name[:-7])
            if original and original['status']=='completed' and run['status']=='completed':
                if [(r['request_id'],r['prompt_sha256']) for r in original['rows']] != [(r['request_id'],r['prompt_sha256']) for r in run['rows']]:
                    raise ValueError('original repeat manifest mismatch')
                original_repeats.append({'original':name[:-7],'repeat':name,
                    'elapsed_change_percent':100*(run['sum_generate_ms']/original['sum_generate_ms']-1),
                    'cached_tokens_identical':run['cached_tokens']==original['cached_tokens']})
        if not name.endswith('-stock'):
            continue
        other = runs.get(name[:-5]+'reuse2')
        if not other or run['status']!='completed' or other['status']!='completed':
            continue
        keys=lambda r:[(x['request_id'],x['prompt_sha256'],x['prompt_tokens']) for x in r['rows']]
        if keys(run)!=keys(other):
            raise ValueError('pair manifests/tokens differ: '+name)
        pairs.append({'stock':name,'reuse2':name[:-5]+'reuse2',
            'cached_token_gain':None if run['cached_tokens'] is None or other['cached_tokens'] is None else other['cached_tokens']-run['cached_tokens'],
            'elapsed_change_percent':100*(other['sum_generate_ms']/run['sum_generate_ms']-1),
            'output_token_hash_matches':sum(a['output_token_sha256']==b['output_token_sha256'] for a,b in zip(run['rows'],other['rows'])),
            'n':run['n']})
    for run in runs.values():
        run.pop('rows')
    scenes={}
    for pair in pairs:
        scene=pair['stock'].split('-',1)[1].rsplit('-',1)[0]
        scenes.setdefault(scene,[]).append(pair)
    repeated={scene:{'completed_pairs':len(group),
        'median_elapsed_change_percent':statistics.median(x['elapsed_change_percent'] for x in group),
        'elapsed_change_percent_range':[min(x['elapsed_change_percent'] for x in group),max(x['elapsed_change_percent'] for x in group)],
        'cached_token_gains':[x['cached_token_gain'] for x in group]}
        for scene,group in scenes.items()}
    return {'runs':runs,'pairs':pairs,'original_repeats':original_repeats,'repeat_summary':repeated,'limitations':[
        'Serial offline measurements are not TTFT or online throughput.',
        'Explicit small KV budgets study pressure; do not represent the GPU default cache.',
        'Report variation between independent repeats; individual requests are correlated.',
        'Curated LooGLE excerpts are not production traces or QA accuracy evaluations.',
        'Output hashes are a smoke check; cache reuse can cause numerical differences.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    a=p.parse_args()
    root=Path(a.input)
    (root/'analysis.json').write_text(json.dumps(analyze(root),indent=2)+'\n')
