"""Send a Git-identified source snapshot; recover and verify each completed fold.

Run with .venv/bin/python scripts/run_foundation_colab.py --model tabfm.
This owns only the newly allocated named session and always releases it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from datafest.lineage import file_record, verify_manifest_integrity, verify_file_records, write_json, sha256_file


def command(args, **kwargs):
    return subprocess.run(args,check=True,text=True,**kwargs)


def source_snapshot(destination: Path) -> str:
    """Create an unattached Git commit; preserve the user's index and branch."""
    with tempfile.TemporaryDirectory() as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/"index"))
        command(["git","read-tree","HEAD"],env=env)
        paths=[str(p) for folder in ("src","tests","scripts") for p in Path(folder).rglob("*.py")]
        paths += ["pyproject.toml","uv.lock"]
        command(["git","add","--",*paths],env=env)
        tree=command(["git","write-tree"],env=env,capture_output=True).stdout.strip()
        commit=command(["git","-c","user.name=Datafest experiment","-c","user.email=datafest@localhost",
                        "commit-tree",tree,"-p","HEAD","-m","Snapshot for rolling foundation experiments"],
                        capture_output=True).stdout.strip()
    command(["git","archive","--format=tar.gz","-o",str(destination),commit,
             "src","tests","scripts","pyproject.toml","uv.lock"])
    return commit


def run(model, gpu, session, existing=False, reuse_runtime=False, fold=None):
    root=Path("experiments/diagnostics/04_foundation")
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="datafest-colab-") as temp:
        temp=Path(temp)
        commit=source_snapshot(temp/"source.tar.gz")
        import shutil
        snapshot=root/"source_snapshots"/f"{commit}.tar.gz"
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(temp/"source.tar.gz",snapshot)
        write_json(root/"source_snapshot.json",{"git_commit":commit,"archive":file_record(snapshot)})
        requirements=Path("experiments/diagnostics/foundation-requirements.txt")
        if not requirements.exists():
            raise RuntimeError("Missing pinned foundation-requirements.txt")
        bundle=temp/"bundle.tar.gz"
        with tarfile.open(bundle,"w:gz") as archive:
            archive.add(temp/"source.tar.gz",arcname="source.tar.gz")
            archive.add(requirements,arcname="foundation-requirements.txt")
            archive.add(root/"source_snapshot.json",arcname=str(root/"source_snapshot.json"))
            archive.add(snapshot,arcname=str(snapshot))
            if not reuse_runtime:
                archive.add("data/train.csv",arcname="data/train.csv")
                for name in ("features.csv.gz","feature_schema.json","split_assignments.csv"):
                    shared=root/model/name
                    if shared.exists(): archive.add(shared,arcname=str(shared))
            # Complete folds stay local. New runtimes receive only code/data;
            # old model weights are not needed to fit later rolling folds.
        created=existing
        try:
            if not existing:
                command(["colab","new","-s",session,"--gpu",gpu])
                created=True
            command(["colab","upload","-s",session,str(bundle),"/content/datafest-bundle.tar.gz"])
            setup=temp/"setup.py"
            setup.write_text('''import os, subprocess, tarfile
from pathlib import Path
def stream(cmd):
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout: print(line,end='',flush=True)
    if proc.wait(): raise RuntimeError('Remote command failed: '+repr(cmd))
root=Path('/content/datafest')
root.mkdir(exist_ok=True)
with tarfile.open('/content/datafest-bundle.tar.gz') as tar: tar.extractall(root,filter='data')
with tarfile.open(root/'source.tar.gz') as tar: tar.extractall(root,filter='data')
stream(['python','-m','pip','install','uv'])
stream(['uv','venv','/content/datafest-env','--python','3.12','--managed-python'])
stream(['uv','pip','install','--python','/content/datafest-env/bin/python','-r',str(root/'foundation-requirements.txt')])
stream(['nvidia-smi'])
''')
            if reuse_runtime:
                code=setup.read_text().split("stream(['python','-m','pip','install','uv'])")[0]
                setup.write_text(code)
            command(["colab","exec","-s",session,"--timeout","1800","-f",str(setup)])
            if model in ("tabpfn3","tabpfn35") and os.environ.get("TABPFN_TOKEN"):
                # The credential is outside the Git snapshot, logs and command
                # arguments. The remote wrapper consumes and deletes this file.
                token_file = temp/"tabpfn.token"
                token_file.touch(mode=0o600)
                token_file.write_text(os.environ["TABPFN_TOKEN"])
                command(["colab","upload","-s",session,str(token_file),"/content/datafest-tabpfn.token"])
                token_file.unlink()
            for month in (202609,202610,202611):
                if fold is not None and month!=fold:
                    continue
                local_state=root/model/str(month)/"state.json"
                if local_state.exists():
                    prior=json.loads(local_state.read_text())
                    if prior.get("status")=="complete":
                        errors=verify_file_records(prior["artifacts"])
                        if errors: raise RuntimeError(f"Local complete fold corruption: {errors}")
                        if prior["identity"]["train_sha256"]!=sha256_file("data/train.csv"):
                            raise RuntimeError("Completed fold source differs")
                        for name,digest in prior["identity"]["code_sha256"].items():
                            if sha256_file(Path("src/datafest")/name)!=digest:
                                raise RuntimeError("Completed fold code differs")
                        print(f"Reusing verified local fold {month}",flush=True)
                        continue
                remote=temp/f"fold_{month}.py"
                worker=temp/f"worker_{month}.py"
                worker.write_text(f'''import os, sys
from pathlib import Path
os.chdir('/content/datafest')
sys.path.insert(0,'/content/datafest/src')
from datafest.{"foundation35" if model=="tabpfn35" else "foundation"} import run_foundation
result=run_foundation({model!r},Path('data'),Path('experiments/diagnostics/04_foundation'),'cuda',{month})
print(result,flush=True)
''')
                remote.write_text(f'''import os, tarfile, subprocess, traceback
from pathlib import Path
os.chdir('/content/datafest')
token_path=Path('/content/datafest-tabpfn.token')
if token_path.exists():
    private_token=token_path.read_text().strip()
    os.environ['TABPFN_TOKEN']=private_token
    token_path.unlink()
Path('/content/datafest-worker.py').write_text({worker.read_text()!r})
try:
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    proc=subprocess.Popen(['/content/datafest-env/bin/python','/content/datafest-worker.py'],env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout: print(line,end='',flush=True)
    print('worker_exit_code:',proc.wait(),flush=True)
except Exception:
    traceback.print_exc()
finally:
    print('fold worker finished; ready for streaming recovery',flush=True)
''')
                # Recovery is attempted even when execution reports failure.
                subprocess.run(["colab","exec","-s",session,"--timeout","3600","-f",str(remote)],check=False)
                cli_python=Path(shutil.which("colab")).read_text().splitlines()[0].removeprefix("#!")
                command([cli_python,"scripts/recover_colab_fold.py","--session",session,"--model",model,"--month",str(month)])
                for path in (root/model).rglob("manifest.json"):
                    errors=verify_manifest_integrity(json.loads(path.read_text()))
                    if errors:
                        raise RuntimeError(f"Recovered integrity errors at {path}: {errors}")
                state=json.loads((root/model/str(month)/"state.json").read_text())
                write_json(root/f"recovery_{model}_{month}.json",{"git_commit":commit,"month":month,
                    "transport":"streaming authenticated files; unchanged files reused by SHA-256",
                    "fold_state":state["status"],"verified":True})
                if state["status"]!="complete":
                    raise RuntimeError(f"Remote fold failed: {state.get('error_type')}: {state.get('error')}")
            # Earlier complete folds need not reside on a replacement VM.
            # Assemble the rolling result from the verified local artifacts.
            import pandas as pd
            from datafest.diagnostics import atomic_csv, manifest, metrics, probability_frame
            target=root/model
            available=[]
            for month in (202609,202610,202611):
                state_path=target/str(month)/"state.json"
                if state_path.exists() and json.loads(state_path.read_text()).get("status")=="complete":
                    available.append(month)
            if len(available)!=3:
                write_json(target/"state.json",{"status":"partial","completed_months":available})
                config=json.loads((target/"config.json").read_text())
                manifest(target,config,[Path("data/train.csv")],[target/"feature_schema.json",target/"features.csv.gz"])
                print(f"Recovered complete folds: {available}",flush=True)
                return
            source=pd.read_csv("data/train.csv")
            frames=[]
            for month in (202609,202610,202611):
                frame=pd.read_csv(target/str(month)/"predictions.csv")
                expected=source.loc[source.mes.eq(month),["id_cliente","mes","objetivo"]].reset_index(drop=True)
                expected.objetivo=expected.objetivo.astype(int)
                pd.testing.assert_frame_equal(frame[expected.columns],expected,check_dtype=False)
                probability_frame(expected,frame.prediccion)
                frames.append(frame)
            combined=pd.concat(frames,ignore_index=True)
            atomic_csv(combined,target/"rolling_predictions.csv")
            write_json(target/"metrics.json",metrics(combined))
            write_json(target/"state.json",{"status":"complete","folds":[202609,202610,202611]})
            config=json.loads((target/"config.json").read_text())
            manifest(target,config,[Path("data/train.csv")],[target/"feature_schema.json",target/"features.csv.gz"])
        finally:
            if created:
                # A reclaimed VM must not mask the original execution or
                # recovery error with a second failure during cleanup.
                subprocess.run(["colab","stop","-s",session],check=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["tabpfn3","tabpfn35","tabfm"],required=True)
    parser.add_argument("--gpu",default="T4")
    parser.add_argument("--session",default="datafest-foundation")
    parser.add_argument("--existing-session",action="store_true",help="Use and release this task's already allocated session")
    parser.add_argument("--reuse-runtime",action="store_true",help="Reuse the prepared environment and data on this task's session")
    parser.add_argument("--fold",type=int,choices=[202609,202610,202611],help="Use a fresh VM for just one fold, then recover and release it")
    args=parser.parse_args()
    run(args.model,args.gpu,args.session,args.existing_session,args.reuse_runtime,args.fold)
