"""Recover completed fold files directly, without a memory-sized tar download."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from datafest.lineage import file_record, sha256_file, verify_manifest_integrity, write_json
from colab_stream_download import download


def restore_compressed(source, target, expected_sha256):
    """Publish only the exact original bytes, regardless of transport encoding."""
    target=Path(target)
    temporary=target.with_name(target.name+'.partial')
    with gzip.open(source,'rb') as compressed,temporary.open('wb') as output:
        shutil.copyfileobj(compressed,output,1024*1024)
    if sha256_file(temporary)!=expected_sha256:
        raise RuntimeError('Decompressed artifact checksum differs; target unchanged')
    os.replace(temporary,target)


def prepare_compression(session, records, model, month):
    selected=[r for r in records if r['bytes']>=32*1024*1024]
    if not selected:
        return {}
    paths=[r['path'] for r in selected]
    code=f'''import gzip,shutil
from pathlib import Path
for name in {paths!r}:
    source=Path('/content/datafest')/name
    target=Path('/content/datafest-transfer-compressed')/(name+'.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as reader,target.open('wb') as writer:
        with gzip.GzipFile(filename='',mode='wb',fileobj=writer,compresslevel=1,mtime=0) as gz:
            shutil.copyfileobj(reader,gz,1024*1024)
    print('Compressed',name,source.stat().st_size,target.stat().st_size,flush=True)
'''
    with tempfile.TemporaryDirectory() as temp:
        script=Path(temp)/'compress.py'
        script.write_text(code)
        try:
            result=subprocess.run(['colab','exec','-s',session,'--timeout','900','-f',str(script)],
                                  capture_output=True,text=True,timeout=960)
        except subprocess.TimeoutExpired:
            print('Transport compression timed out; using original files',flush=True)
            return {}
    log=re.sub(r'colab-runtime-proxy-token=[^&\s\x27\x22)]+',
               'colab-runtime-proxy-token=[REDACTED]',result.stdout+result.stderr)
    if os.environ.get('TABPFN_TOKEN'):
        log=log.replace(os.environ['TABPFN_TOKEN'],'[REDACTED]')
    audit=Path('experiments/diagnostics/04_foundation/attempts')
    audit.mkdir(parents=True,exist_ok=True)
    (audit/f'compression_{model}_{month}.log').write_text(log)
    if result.returncode:
        print('Transport compression unavailable; using original files',flush=True)
        return {}
    print('Lossless transport compression ready',flush=True)
    return {name:'/content/datafest-transfer-compressed/'+name+'.gz' for name in paths}


def recover(session,model,month):
    root=Path("experiments/diagnostics/04_foundation")/model
    root.mkdir(parents=True,exist_ok=True)
    parent=root/"manifest.json"
    download(session,"/content/datafest/"+str(parent),str(parent))
    manifest=json.loads(parent.read_text())
    missing=[]
    for record in manifest["outputs"]:
        # The default TabPFN-3.5 checkpoint is already preserved separately.
        # Its serialized live estimator is redundant for resume and too large
        # to recover reliably before Colab reclaims the session.
        if model=="tabpfn35" and Path(record["path"]).name=="fitted_estimator.pkl":
            continue
        path=Path(record["path"])
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected artifact path")
        if model=="tabpfn35" and path.parent.name=="checkpoint" and not path.is_file():
            for candidate in root.glob(f"*/checkpoint/{path.name}"):
                if candidate.is_file() and candidate.stat().st_size==record["bytes"] and sha256_file(candidate)==record["sha256"]:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(candidate,path)
                    print(f"Reused verified shared checkpoint for {month}",flush=True)
                    break
            if path.is_file():
                continue
        if path.is_file() and path.stat().st_size==record["bytes"] and sha256_file(path)==record["sha256"]:
            continue
        missing.append(record)
    # Recover probabilities and completion metadata before large weights.
    small=[r for r in missing if r['bytes']<32*1024*1024]
    large=[r for r in missing if r['bytes']>=32*1024*1024]
    for record in small:
        path=Path(record['path'])
        download(session,'/content/datafest/'+str(path),str(path))
        if sha256_file(path)!=record['sha256']:
            raise RuntimeError(f'Recovered checksum mismatch: {path}')
    compressed=prepare_compression(session,large,model,month)
    for record in large:
        path=Path(record['path'])
        if str(path) in compressed:
            transport=path.with_name(path.name+'.transport.gz')
            download(session,compressed[str(path)],str(transport))
            restore_compressed(transport,path,record['sha256'])
            transport.unlink()
        else:
            download(session,"/content/datafest/"+str(path),str(path))
        if sha256_file(path)!=record["sha256"]:
            raise RuntimeError(f"Recovered checksum mismatch: {path}")
    leaf=root/str(month)/"manifest.json"
    if model=="tabpfn35":
        state_path=root/str(month)/"state.json"
        state=json.loads(state_path.read_text())
        state["artifacts"]=[r for r in state["artifacts"]
                            if Path(r["path"]).name!="fitted_estimator.pkl"]
        write_json(state_path,state)
        download(session,"/content/datafest/"+str(leaf),str(leaf))
        for manifest_path in (parent,leaf):
            document=json.loads(manifest_path.read_text())
            document["outputs"]=[r for r in document["outputs"]
                                 if Path(r["path"]).name!="fitted_estimator.pkl"]
            for index,record in enumerate(document["outputs"]):
                if Path(record["path"]).resolve()==state_path.resolve():
                    document["outputs"][index]=file_record(state_path)
            write_json(manifest_path,document)
    else:
        download(session,"/content/datafest/"+str(leaf),str(leaf))
    for path in (parent,leaf):
        errors=verify_manifest_integrity(json.loads(path.read_text()))
        if errors: raise RuntimeError(f"Recovered integrity errors: {errors}")
    state=json.loads((root/str(month)/"state.json").read_text())
    if state["status"]!="complete":
        raise RuntimeError(f"Fold was not complete: {state['status']}")
    print(f"Fold {month} recovered and verified",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--session",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--month",required=True,type=int)
    a=p.parse_args()
    recover(a.session,a.model,a.month)
