"""Run either explicitly selected TabPFN version using a private root .env."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import shutil

from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from datafest.lineage import file_record,write_json
from run_foundation_colab import source_snapshot


def load_private_token():
    values=dotenv_values(ROOT/'.env')
    names=('TABPFN_TOKEN','TABPFN_API_KEY','PRIORLABS_API_KEY','API_KEY','api_key')
    name=next((n for n in names if values.get(n)),None)
    if name is None:
        present=[n for n,v in values.items() if v]
        if len(present)==1:
            name=present[0]
    token=values[name] if name else os.environ.get('TABPFN_TOKEN')
    if not token:
        raise RuntimeError('No unique API key in root .env or TABPFN_TOKEN; values omitted')
    os.environ['TABPFN_TOKEN']=token.strip()
    if (ROOT/'.env').exists():
        (ROOT/'.env').chmod(0o600)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['tabpfn3','tabpfn35'],required=True)
    parser.add_argument('--fold',type=int,choices=[202609,202610,202611])
    args=parser.parse_args()
    os.chdir(ROOT)
    load_private_token()
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    root=Path('experiments/diagnostics/04_foundation')
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'source.tar.gz'
        commit=source_snapshot(archive)
        target=root/'source_snapshots'/f'{commit}.tar.gz'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(archive,target)
        write_json(root/'source_snapshot.json',{'git_commit':commit,'archive':file_record(target)})
    if args.model=='tabpfn3':
        from datafest.foundation import run_foundation
    else:
        from datafest.foundation35 import run_foundation
    try:
        print(run_foundation(args.model,Path('data'),root,'cuda',args.fold),flush=True)
    finally:
        cache=Path.home()/'.cache/tabpfn/auth_token'
        if cache.exists():
            cache.chmod(0o600)


if __name__=='__main__':
    main()
