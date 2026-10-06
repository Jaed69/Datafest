# Neural tabular candidates (provenance for N1 / N2)

These scripts produced the neural out-of-fold predictions (`oof_neural.csv`) and the December predictions
(`test_neural.csv`) that are copied, unchanged, into `experiments/ensembles/<run_id>/neural/`.
They are kept exactly as they were run. N1 = pytabkit `RealMLP_TD_Classifier`, N2 = `TabM_D_Classifier`,
default hyper-parameters, `val_metric_name="1-auc_ovr"`, seeds 42/43/44 averaged by mean probability,
features = the repo `D_absolute` matrix. The `*rank` variants (`Drank`) are exploratory and were not selected.

## Environment (exact)

- OS: Windows 11, GPU NVIDIA RTX 5060 Laptop (CUDA 12.8 wheels), Python 3.12.13 (uv-managed CPython).
- `requirements_frozen.txt` (this folder) is the complete frozen environment, e.g. `torch==2.11.0+cu128`,
  `pytabkit==1.7.3`, `tabm==0.0.3`, `rtdl-num-embeddings==0.0.12`, `pytorch-lightning==2.6.6`,
  `numpy==2.4.4`, `pandas==3.0.2`, `scikit-learn==1.8.0`, `scipy==1.18.1`, `lightgbm==4.6.0`,
  `catboost==1.2.8`, `xgboost==3.4.1`.
- This environment is separate from the repo's uv environment (`pyproject.toml` does not carry torch).

## Required layout

The scripts resolve `out/`, `logs/`, the virtualenv and the repo relative to their own location, so run them
from a working directory `<work>` laid out as below (copy this folder to `<work>/scripts`):

```
<work>/
  .venv/            # Python 3.12 venv built from requirements_frozen.txt
  scripts/          # a copy of scripts/neural/
  out/              # checkpoints (ckpt/, final/), timings, summary.csv; created by the scripts
  logs/             # per-process logs (create it first)
```

`nn_common.py` imports the repo package read-only from `DATAFEST_REPO` (default is the author's local path,
override it) and reads `data/train.csv` and `data/test.csv` from there.

## Commands

```bash
# 1. environment (from <work>)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe \
  --extra-index-url https://download.pytorch.org/whl/cu128 -r scripts/requirements_frozen.txt
mkdir -p logs
export DATAFEST_REPO=/path/to/Datafest   # repo checkout containing data/ and src/

# 2. rolling Sep/Oct/Nov out-of-fold predictions, one process per (fold, seed)
#    (this is what was run: N1 and N2, variants D and Drank, folds 202609/202610/202611, seeds 42/43/44)
NN_THREADS=1 bash scripts/launch_parallel.sh N1 D \
  "202609:42 202609:43 202609:44 202610:42 202610:43 202610:44 202611:42 202611:43 202611:44"
NN_THREADS=1 bash scripts/launch_parallel.sh N2 D \
  "202609:42 202609:43 202609:44 202610:42 202610:43 202610:44 202611:42 202611:43 202611:44"

# 3. assemble OOF (seed average, joined to the repo OOF by (id_cliente, mes)), blends, bootstrap
#    -> out/oof_neural.csv, out/summary.csv, out/results.md
.venv/Scripts/python.exe scripts/evaluate.py

# 4. December fit on all Jan-Nov rows, one process per (model, seed), then assemble
STAGGER=30 bash scripts/launch_final.sh D "N1:42 N1:43 N1:44 N2:42 N2:43 N2:44"
.venv/Scripts/python.exe scripts/final_fit_neural.py --assemble --variant D   # writes <work>/test_neural.csv

# optional diagnostics
.venv/Scripts/python.exe scripts/seed_noise.py        # single-seed spread
.venv/Scripts/python.exe scripts/test_agreement.py    # December agreement vs final GBDT submissions
```

Notes: the N1 fits take roughly 17-30 minutes each on the RTX 5060 Laptop (N2 about 1.5-6 minutes), so the
rolling runs were launched with several processes in parallel (`launch_parallel.sh`, staggered to avoid CUDA/RAM
initialisation failures). GPU training is not bit-for-bit deterministic across hardware or driver versions, which
is why the predictions are stored and copied into the ensemble run instead of being regenerated on demand.
`evaluate.py` reads `experiments/hypotheses/20261006T100749205414Z_a714ebcd/oof_predictions.csv`, which is
gitignored; regenerate it with `uv run datafest hypotheses` if absent.
