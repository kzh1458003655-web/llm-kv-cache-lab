"""Stream fixed, scheduled requests; public results contain hashes, not text."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[math.floor(fraction * (len(values) - 1))] if values else None


async def metrics(session, base_url):
    try:
        async with session.get(base_url + '/metrics', timeout=10) as response:
            response.raise_for_status()
            text = await response.text()
        return {'status': 'ok', 'lines': [line for line in text.splitlines()
            if not line.startswith('#') and line.startswith('vllm:') and
            any(word in line for word in ('prefix_cache', 'preemption', 'num_requests', 'kv_cache_usage'))]}
    except Exception as exc:
        return {'status': 'failed', 'error_type': type(exc).__name__}


async def stream_one(session, base_url, row, output_tokens, timeout):
    start = time.perf_counter()
    first_token = last_token = first_text = None
    ids = []
    text_digest = hashlib.sha256()
    usage = None
    saw_done = False
    result = {'request_id': row['request_id'], 'doc_id': row['doc_id'],
              'prompt_sha256': row['prompt_sha256'],
              'input_token_ids_sha256': row['input_token_ids_sha256'],
              'expected_prompt_tokens': row['expected_prompt_tokens'],
              'status': 'failed'}
    payload = {'model': 'thesis-9b', 'prompt': row['prompt_token_ids'],
               'request_id': 'measure-' + row['request_id'],
               'max_tokens': output_tokens, 'temperature': 0, 'seed': 0,
               'ignore_eos': True, 'add_special_tokens': False,
               'return_token_ids': True, 'stream_interval': 1,
               'stream': True, 'stream_options': {'include_usage': True}}
    try:
        async with session.post(base_url + '/v1/completions', json=payload, timeout=timeout) as response:
            result['http_status'] = response.status
            response.raise_for_status()
            data_lines = []
            async for raw in response.content:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line.startswith('data:'):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    data = '\n'.join(data_lines)
                    data_lines.clear()
                    if data == '[DONE]':
                        saw_done = True
                        break
                    chunk = json.loads(data)
                    if chunk.get('error'):
                        raise ValueError('server returned stream error')
                    now = time.perf_counter()
                    for choice in chunk.get('choices', []):
                        delta = choice.get('token_ids') or []
                        if delta:
                            first_token = first_token or now
                            last_token = now
                            ids.extend(delta)
                        text = choice.get('text') or ''
                        if text:
                            first_text = first_text or now
                            text_digest.update(text.encode())
                    if chunk.get('usage') is not None:
                        usage = chunk['usage']
        if not saw_done or first_token is None or usage is None:
            raise ValueError('missing DONE, streamed token IDs, or usage')
        if usage['prompt_tokens'] != row['expected_prompt_tokens']:
            raise ValueError('server input token count differs from fixed input')
        if len(ids) != output_tokens or usage['completion_tokens'] != output_tokens:
            raise ValueError('output length differs from fixed token budget')
        result['status'] = 'ok'
    except Exception as exc:
        # Error bodies may echo a prompt. Publish the type only.
        result['error_type'] = type(exc).__name__
    end = time.perf_counter()
    result.update({'elapsed_ms': (end - start) * 1000,
        'ttft_ms': (first_token - start) * 1000 if first_token else None,
        'first_text_ms': (first_text - start) * 1000 if first_text else None,
        'tpot_ms': (last_token - first_token) * 1000 / (len(ids) - 1)
                   if first_token and len(ids) > 1 else None,
        'output_tokens': len(ids), 'usage': usage,
        'output_token_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        'output_text_sha256': text_digest.hexdigest()})
    return result


async def run_client(base_url, rows, concurrency, rate, output_tokens, output_dir,
                     timeout_seconds=300, warmup=True, reset_cache=True):
    import aiohttp
    if concurrency <= 0 or rate <= 0 or not math.isfinite(rate):
        raise ValueError('concurrency and finite rate must be positive')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    # One extra connection lets metrics sampling run while request slots fill.
    connector = aiohttp.TCPConnector(limit=concurrency + 1)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup:
            for row in rows[:2]:
                warmup_row = {**row, 'request_id': 'warmup-' + row['request_id']}
                result = await stream_one(session, base_url, warmup_row, output_tokens, timeout)
                if result['status'] != 'ok':
                    (out / 'warmup-failure.json').write_text(json.dumps(result, indent=2) + '\n')
                    raise RuntimeError('streaming warmup failed; see warmup-failure.json')
        if reset_cache:
            async with session.post(base_url + '/reset_prefix_cache', timeout=30) as response:
                response.raise_for_status()
                await response.read()
        before = await metrics(session, base_url)
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(concurrency)
        active = peak = 0
        results = []
        samples = []
        finished = asyncio.Event()

        async def monitor():
            while not finished.is_set():
                snapshot = await metrics(session, base_url)
                values = {}
                for line in snapshot.get('lines', []):
                    match = re.match(r'vllm:(num_requests_running|num_requests_waiting)(?:\{.*\})?\s+([0-9.eE+-]+)', line)
                    if match:
                        values[match[1]] = values.get(match[1], 0) + float(match[2])
                samples.append({'offset_s': time.perf_counter() - start, **values})
                try:
                    await asyncio.wait_for(finished.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

        async def request(row):
            nonlocal active, peak
            scheduled = row['arrival_units'] / rate
            await asyncio.sleep(max(0, start + scheduled - time.perf_counter()))
            reached = time.perf_counter()
            async with semaphore:
                sent = time.perf_counter()
                active += 1
                peak = max(peak, active)
                try:
                    result = await stream_one(session, base_url, row, output_tokens, timeout)
                finally:
                    active -= 1
            result.update({'scheduled_offset_s': scheduled,
                'actual_send_offset_s': sent - start,
                'arrival_lateness_ms': (reached - start - scheduled) * 1000,
                'client_wait_ms': (sent - reached) * 1000,
                'arrival_to_first_token_ms': (sent - start - scheduled) * 1000 + result['ttft_ms']
                    if result['ttft_ms'] is not None else None})
            results.append(result)
            with (out / 'requests.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, separators=(',', ':')) + '\n')

        monitor_task = asyncio.create_task(monitor())
        try:
            await asyncio.gather(*(request(row) for row in rows))
        finally:
            finished.set()
            await monitor_task
        duration = time.perf_counter() - start
        after = await metrics(session, base_url)
    ok = [row for row in results if row['status'] == 'ok']
    summary = {'request_count': len(rows), 'success_count': len(ok),
        'error_count': len(rows) - len(ok), 'duration_s': duration,
        'request_rate': rate, 'concurrency_limit': concurrency, 'peak_client_active': peak,
        'max_server_running': max((sample.get('num_requests_running', 0) for sample in samples), default=0),
        'max_server_waiting': max((sample.get('num_requests_waiting', 0) for sample in samples), default=0),
        'server_activity_samples': samples,
        'output_tokens_per_s': sum(row['output_tokens'] for row in ok) / duration,
        'throughput_note': 'achieved throughput at this arrival rate; not maximum serving capacity',
        'timing_note': 'TTFT = first streamed token-ID chunk minus send; TPOT = first-to-last token-ID chunk interval / (N-1)',
        'percentile_definition': 'sorted[floor(p*(n-1))]',
        'metrics_before': before, 'metrics_after': after}
    for name in ('ttft_ms', 'tpot_ms', 'elapsed_ms', 'arrival_to_first_token_ms', 'client_wait_ms'):
        values = [row[name] for row in ok if row[name] is not None]
        summary[name] = {f'p{int(p*100)}': percentile(values, p) for p in (.5, .95, .99)}
    cached = [(row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for row in ok]
    summary['cached_tokens'] = sum(cached) if cached and all(value is not None for value in cached) else None
    summary['input_tokens'] = sum(row['usage']['prompt_tokens'] for row in ok)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
