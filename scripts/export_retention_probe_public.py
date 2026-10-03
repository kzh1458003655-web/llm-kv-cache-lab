"""Build a public, text-free projection of one retention diagnostic."""
import argparse
import json
from pathlib import Path
import re


REQUEST_ALLOW = ('request_id', 'input_token_ids_sha256', 'expected_prompt_tokens',
    'status', 'http_status', 'elapsed_ms', 'ttft_ms', 'tpot_ms', 'output_tokens',
    'usage', 'output_token_ids_sha256', 'scheduled_offset_s', 'actual_send_offset_s',
    'arrival_lateness_ms', 'client_wait_ms', 'arrival_to_first_token_ms')


def publicize(source: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=False)
    for folder in sorted(source.glob('run-*')):
        record = json.loads((folder / 'run.json').read_text())
        if record.get('status') != 'completed' or record.get('request_count') != 64:
            raise ValueError(f'incomplete run: {folder.name}')
        out = dest / folder.name
        out.mkdir()
        summary=record['client_summary']
        public_summary={k:summary[k] for k in ('request_count','success_count','error_count',
            'duration_s','request_rate','concurrency_limit','peak_client_active',
            'max_server_running','max_server_waiting','server_activity_metrics_available',
            'cached_tokens','input_tokens') if k in summary}
        for metric in ('ttft_ms','tpot_ms','elapsed_ms','arrival_to_first_token_ms','client_wait_ms'):
            if metric in summary: public_summary[metric]=summary[metric]
        public_run = {k: record[k] for k in ('status','mode','policy','settings',
            'model_revision','request_count','measurement_reset',
            'trace_diagnostic','generated_tokens_per_request','startup_elapsed_s',
            'total_elapsed_s') if k in record}
        public_run['client_summary']=public_summary
        runtime=record['runtime_profile']
        public_run['runtime_profile']={'versions':runtime['versions'], 'gpu':{
            k:runtime['gpu'][k] for k in ('name','memory_mib','driver')}}
        (out/'run.json').write_text(json.dumps(public_run,indent=2)+'\n')
        rows=[json.loads(line) for line in (folder/'client/requests.jsonl').read_text().splitlines()]
        if len(rows)!=64 or any(r.get('status')!='ok' for r in rows):
            raise ValueError(f'failed request in {folder.name}')
        with (out/'requests.jsonl').open('x') as stream:
            for r in rows:
                stream.write(json.dumps({k:r[k] for k in REQUEST_ALLOW if k in r},separators=(',',':'))+'\n')
        evidence=out/'evidence'; evidence.mkdir()
        for name in ('cache-groups','engine-active'):
            paths=list((folder/'evidence').glob(name+'-*.json'))
            if not paths: raise ValueError(f'missing {name}: {folder.name}')
            value=json.loads(paths[0].read_text())
            value.pop('pid',None)
            if name=='engine-active':
                (evidence/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
            else:
                value.pop('tensor_descriptors',None)
                value['groups']=[{k:g[k] for k in ('spec','block_size','page_size_bytes') if k in g} for g in value['groups']]
                (evidence/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        ev=[]
        source_events=[json.loads(line) for line in (folder/'diagnostic-events.jsonl').read_text().splitlines()]
        clock_origin=min((e['monotonic_ns'] for e in source_events if 'monotonic_ns' in e),default=0)
        for item in source_events:
            if any(k in item for k in ('prompt','output_text','prompt_token_ids')):
                raise ValueError('forbidden input/output text in events')
            if 'request_id' in item:
                rid=item.get('request_id','')
                m=re.search(r'(segment\d{3}-[a-zA-Z0-9]+)',rid)
                if m:item['request_id']=m.group(1)
                else:item.pop('request_id',None)
            if 'monotonic_ns' in item:
                item['relative_ns']=item.pop('monotonic_ns')-clock_origin
            ev.append(item)
        with (out/'diagnostic-events.jsonl').open('x') as stream:
            for item in ev: stream.write(json.dumps(item,separators=(',',':'))+'\n')
    for name in ('plan.json','retention-analysis.json'):
        p=source/name
        if p.exists(): (dest/name).write_bytes(p.read_bytes())
    forbidden=re.compile(r'GPU-[0-9a-f-]{20,}|[A-Z]:\\|/root/|prompt_token_ids|output_text|doc_id|server-private')
    findings=[]
    for path in dest.rglob('*'):
        if path.is_file() and forbidden.search(path.read_text(errors='replace')):
            findings.append(str(path.relative_to(dest)))
    if findings: raise ValueError(f'public privacy scan found: {findings}')
    audit={'runs':8,'requests':512,'failed_requests':0,'privacy_scan_findings':findings,'raw_server_logs_included':False,
           'prompt_text_or_token_arrays_included':False,'gpu_uuid_included':False,
           'absolute_host_paths_included':False}
    (dest/'privacy-audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    publicize(args.source,args.destination)
