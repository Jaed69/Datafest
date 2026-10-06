import argparse
import json
from pathlib import Path

from datafest.lineage import verify_manifest_integrity
from datafest.pipeline import run_pipeline
from datafest.ablation import run_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena, registra y verifica experimentos Datafest")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "verify", "ablation", "diagnostics", "foundation", "final-fit", "hypotheses", "ensemble-search"])
    parser.add_argument("--model", choices=["lightgbm", "catboost", "ensemble-h4b2", "ensemble-h4b2-n1"], default="lightgbm", help="Modelo para final-fit")
    parser.add_argument("--variant", default="D", help="Variante de features para final-fit (A, B, C, D, B_month_index, D_month_index)")
    parser.add_argument("--compare-with", help="Submission previa para calcular Spearman en final-fit")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", help="Manifiesto de corrida para el comando verify")
    parser.add_argument("--baseline", default="experiments/ablation/runs/20261001T051935820299Z_7e1c7547")
    parser.add_argument("--foundation-model", choices=["tabpfn3", "tabfm"], default="tabpfn3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replicates", type=int, default=2000, help="Bootstrap resamples para hypotheses")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1, help="Ronda de hypotheses (2 = features + GBDT)")
    parser.add_argument("--trials-lgbm", type=int, default=60, help="Optuna trials de LightGBM (round 2)")
    parser.add_argument("--trials-catboost", type=int, default=40, help="Optuna trials de CatBoost (round 2)")
    parser.add_argument("--cache-dir", help="Checkpoints por candidato para reanudar round 2 (gitignored)")
    parser.add_argument("--neural-dir", default="experiments/neural", help="Directorio con oof_neural.csv/test_neural.csv/results.md (ensemble-search)")
    parser.add_argument("--round1-oof", default="experiments/hypotheses/20261006T100749205414Z_a714ebcd/oof_predictions.csv")
    parser.add_argument("--round2-oof", default="experiments/hypotheses/20261006T143722753859Z_a714ebcd_r2/oof_predictions.csv")
    parser.add_argument("--h4b2-members", default="experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/member_predictions.csv")
    parser.add_argument("--current-submission", default="experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/submission.csv")
    parser.add_argument("--neural-test", default="experiments/ensembles/20261006T173753810737Z_a714ebcd_ensemble-search/neural/test_neural.csv", help="Predicciones N1 de diciembre (final-fit ensemble-h4b2-n1)")
    parser.add_argument("--neural-seeds", help="Directorio con N1_D_s<seed>.npy por semilla (final-fit ensemble-h4b2-n1)")
    parser.add_argument("--provenance", action="append", default=[], help="Evidencia de la decision a registrar (repetible)")
    parser.add_argument("--stack-step", type=float, default=0.1, help="Paso de la grilla simplex del stacking (ensemble-search)")
    parser.add_argument("--fold", type=int, choices=[202609, 202610, 202611])
    args = parser.parse_args()
    if args.command == "final-fit":
        if args.model == "ensemble-h4b2-n1":
            from datafest.final_ensemble_n1 import run_final_ensemble_n1
            result = run_final_ensemble_n1(
                data_dir=args.data_dir, experiment_root=Path(args.experiment_root) / "final",
                h4b2_members=args.h4b2_members, neural_test=args.neural_test,
                neural_seeds_dir=args.neural_seeds, compare_with=args.compare_with or args.current_submission,
                provenance=args.provenance,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.model == "ensemble-h4b2":
            from datafest.final_ensemble import run_final_ensemble
            result = run_final_ensemble(
                data_dir=args.data_dir, experiment_root=Path(args.experiment_root) / "final",
                seed=args.seed, compare_with=args.compare_with,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        from datafest.final_fit import run_final_fit
        result = run_final_fit(
            args.model, args.variant, data_dir=args.data_dir,
            experiment_root=Path(args.experiment_root) / "final", seed=args.seed,
            compare_with=args.compare_with,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "ensemble-search":
        from datafest.ensemble_search import run_ensemble_search
        result = run_ensemble_search(
            args.data_dir, Path(args.experiment_root) / "ensembles", round1_oof=args.round1_oof,
            round2_oof=args.round2_oof, neural_dir=args.neural_dir, h4b2_members=args.h4b2_members,
            current_submission=args.current_submission, replicates=args.replicates, seed=args.seed, step=args.stack_step,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "hypotheses" and args.round == 2:
        from datafest.round2 import run_round2
        result = run_round2(args.data_dir, Path(args.experiment_root) / "hypotheses", seed=args.seed,
                            replicates=args.replicates, trials_lgbm=args.trials_lgbm,
                            trials_catboost=args.trials_catboost, cache_dir=args.cache_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "hypotheses":
        from datafest.hypotheses import run_hypotheses
        result = run_hypotheses(args.data_dir, Path(args.experiment_root) / "hypotheses",
                                seed=args.seed, replicates=args.replicates)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "diagnostics":
        from datafest.diagnostics import run_diagnostics
        result = run_diagnostics(args.data_dir, args.baseline, Path(args.experiment_root)/"diagnostics")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "foundation":
        from datafest.foundation import run_foundation
        result = run_foundation(args.foundation_model, Path(args.data_dir),
                                Path(args.experiment_root)/"diagnostics"/"04_foundation", args.device, args.fold)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
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

    if args.command == "ablation":
        result = run_ablation(
            data_dir=args.data_dir,
            experiment_root=args.experiment_root + "/ablation",
            seed=args.seed,
        )
        print(json.dumps({
            "batch_id": result["batch_id"],
            "runs": len(result["runs"]),
            "manifest": result["manifest"]["path"],
            "summary": result["summary"]["path"],
        }, ensure_ascii=False, indent=2))
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
