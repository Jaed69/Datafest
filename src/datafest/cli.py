import argparse
import json
from pathlib import Path

from datafest.lineage import verify_manifest_integrity
from datafest.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena, registra y verifica experimentos Datafest")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "verify"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", help="Manifiesto de corrida para el comando verify")
    args = parser.parse_args()
    if args.command == "verify":
        if not args.manifest:
            parser.error("verify requiere --manifest")
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        errors = verify_manifest_integrity(manifest)
        if errors:
            print(json.dumps({"verified": False, "errors": errors}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        print(json.dumps({"verified": True, "run_id": manifest.get("run_id")}, indent=2))
        return

    result = run_pipeline(
        data_dir=args.data_dir,
        experiment_root=args.experiment_root,
        split_root=args.split_root,
        processed_root=args.processed_root,
        seed=args.seed,
    )
    print(json.dumps({
        "batch_id": result["batch_id"],
        "split_id": result["split_id"],
        "winner_assessment": result["winner_assessment"],
        "runs": len(result["runs"]),
        "index": f"{args.experiment_root}/index.json",
    }, ensure_ascii=False, indent=2))
