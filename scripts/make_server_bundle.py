"""Build an allowlisted private upload tar; never add weights/data to Git."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    root=Path(__file__).resolve().parents[1]
    dest=root/'data/raw/upload/hybrid-server-ready'
    dest.mkdir(parents=True,exist_ok=False)
    files=['kv_cache_lab/__init__.py','kv_cache_lab/events.py','kv_cache_lab/hybrid_retention_prototype.py',
           'scripts/server_benchmark.py','scripts/run_server_round.py',
           'scripts/analyze_server_round.py','scripts/server_workloads.py',
           'scripts/prepare_server_round.py','scripts/verify_server_bundle.py',
           'server-kit/setup.sh','server-kit/README.md','server-kit/preparation-status.json',
           'tests/test_server_workloads.py','tests/test_analyze_server_round.py',
           'tests/test_run_server_round.py']
    for relative in files:
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
    shutil.copytree(root/'models/Qwen3.5-0.8B',dest/'model')
    shutil.copytree(root/'data/raw/server-prepared',dest/'prepared')
    shutil.copytree(root/'data/public/hybrid-prefix-retention/server-prepared-20260930',dest/'audits')
    (dest/'source').mkdir()
    shutil.copyfile(root/'data/raw/loogle-server/shortdep_qa.jsonl',dest/'source/shortdep_qa.jsonl')
    checksums={}
    for path in sorted(dest.rglob('*')):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):
                    h.update(chunk)
            checksums[path.relative_to(dest).as_posix()]=h.hexdigest()
    (dest/'CHECKSUMS.json').write_text(json.dumps(checksums,indent=2)+'\n')
    archive=dest.parent/'hybrid-server-ready.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(dest,arcname=dest.name)
    print(str(archive),archive.stat().st_size,flush=True)


if __name__=='__main__':
    main()
