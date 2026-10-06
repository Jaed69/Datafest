"""Honest ensemble decision on the Sep/Oct/Nov rolling out-of-fold predictions.

The candidate list is fixed a priori (no search beyond it). Every candidate is a
weighted blend of *within-month percentile ranks*, so it only depends on the
ordering each member produces (the metric is Gini). Because only three
validation months exist, the module reports three kinds of evidence:

* fixed candidates: per-month Gini and a paired client bootstrap against the
  champion, Holm-adjusted across the whole family;
* leave-one-month-out (LOMO) procedures: a non-negative simplex rank-weight
  stack and "pick the best candidate on two months" are fitted on two months
  and scored on the third, which is the out-of-selection estimate;
* probability of being best (per month and per bootstrap replicate).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from datafest import __version__
from datafest.diagnostics import _weighted_gini_preparation
from datafest.hypotheses import holm_adjust, within_month_rank
from datafest.lineage import build_run_manifest, sha256_file, write_json
from datafest.metrics import gini_score

KEYS = ("id_cliente", "mes")
MONTHS = (202609, 202610, 202611)
BASELINE = "H4b2"
STACK_NAME = "STACK_LOMO"
BESTOF_NAME = "BESTOF_LOMO"

# Joined-table column -> (source table, source column).
ROUND2_MEMBERS = {"B0": "B0_lgbm_D", "H1b": "H1b_catboost", "catF1": "cat_A_H1b+F1_calendar", "xgb": "xgb_D"}
STACK_MEMBERS = ("B0", "H1b", "catF1", "N1", "xgb")
DECEMBER_MEMBERS = {"B0": "prob_lightgbm", "H1b": "prob_catboost"}  # + N1 from the neural test file
DECEMBER_MONTH = 202612


# ------------------------------------------------------------------------ join
def join_oof(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Join prediction tables on (id_cliente, mes), asserting identical rows and target."""
    if not frames:
        raise ValueError("at least one table is required")
    joined: pd.DataFrame | None = None
    for position, frame in enumerate(frames):
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"table {position} has duplicate (id_cliente, mes) keys")
        frame = frame.sort_values(list(KEYS), kind="stable").reset_index(drop=True)
        if joined is None:
            joined = frame
            continue
        left = pd.MultiIndex.from_frame(joined[list(KEYS)])
        right = pd.MultiIndex.from_frame(frame[list(KEYS)])
        if len(left) != len(right) or not left.equals(right):
            raise ValueError(f"table {position} has a different row set than the first table")
        if "objetivo" in frame.columns and "objetivo" in joined.columns:
            if not np.array_equal(joined["objetivo"].to_numpy(), frame["objetivo"].to_numpy()):
                raise ValueError(f"table {position} has a different objetivo than the first table")
        extra = [c for c in frame.columns if c not in KEYS and c != "objetivo"]
        clash = [c for c in extra if c in joined.columns]
        if clash:
            raise ValueError(f"column clash when joining table {position}: {clash}")
        if "objetivo" in frame.columns and "objetivo" not in joined.columns:
            extra = ["objetivo", *extra]
        joined = pd.concat([joined, frame[extra]], axis=1)
    assert joined is not None
    return joined


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class Candidate:
    """Weighted groups; each group is the within-month rank of the rank-average of its members."""

    name: str
    groups: tuple[tuple[float, tuple[str, ...]], ...]

    @property
    def members(self) -> frozenset[str]:
        return frozenset(m for _, members in self.groups for m in members)


def blend_scores(frame: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    total = sum(weight for weight, _ in candidate.groups)
    if not total > 0:
        raise ValueError(f"candidate {candidate.name}: weights must have a positive sum")
    months = frame["mes"]
    score = np.zeros(len(frame))
    for weight, members in candidate.groups:
        ranks = [within_month_rank(frame[m], months) for m in members]
        group = np.mean(ranks, axis=0)
        if len(members) > 1:
            group = within_month_rank(pd.Series(group), months)
        score += weight / total * group
    return score


def fixed_candidates() -> list[Candidate]:
    """The a-priori candidate list (the first entry is the baseline)."""
    return [
        Candidate("H4b2", ((1.0, ("B0", "H1b")),)),
        Candidate("H4b2+N1", ((1.0, ("B0",)), (1.0, ("H1b",)), (1.0, ("N1",)))),
        Candidate("E1", ((1.0, ("B0", "catF1")),)),
        Candidate("E1+N1", ((1.0, ("B0",)), (1.0, ("catF1",)), (1.0, ("N1",)))),
        Candidate("B0+N1", ((1.0, ("B0",)), (1.0, ("N1",)))),
        Candidate("0.5*rank(H4b2)+0.5*rank(N1)", ((0.5, ("B0", "H1b")), (0.5, ("N1",)))),
    ]


# --------------------------------------------------------------------- metrics
def monthly_ginis(frame: pd.DataFrame, score: np.ndarray, months=MONTHS) -> np.ndarray:
    result = []
    for month in months:
        mask = frame["mes"].eq(month).to_numpy()
        result.append(gini_score(frame.loc[mask, "objetivo"], score[mask]))
    return np.asarray(result)


# ---------------------------------------------------------------------- stack
def simplex_grid(n_members: int, step: float) -> np.ndarray:
    """All non-negative weight vectors on a regular grid that sum to one."""
    units = round(1 / step)
    if n_members < 1 or abs(units * step - 1) > 1e-9:
        raise ValueError("step must divide 1 exactly")
    rows: list[list[int]] = []

    def build(prefix: list[int], remaining: int) -> None:
        if len(prefix) == n_members - 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value)

    build([], units)
    return np.asarray(rows, dtype=float) / units


def fit_rank_weights(ranks: np.ndarray, y: np.ndarray, months: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Simplex weights maximising the mean per-month Gini of ``ranks @ w`` (grid search).

    Ties keep the first grid point in lexicographic order, so the fit is deterministic.
    A coarse grid is deliberate: it is the only regularisation with two fitting months.
    """
    ranks, y, months = np.asarray(ranks, dtype=float), np.asarray(y), np.asarray(months)
    grid = simplex_grid(ranks.shape[1], step)
    masks = [months == m for m in np.unique(months)]
    objective = np.zeros(len(grid))
    for mask in masks:
        scores = ranks[mask] @ grid.T
        objective += np.array([gini_score(y[mask], scores[:, i]) for i in range(len(grid))])
    objective = np.round(objective / len(masks), 12)
    return grid[int(np.argmax(objective))]


def _rank_matrix(frame: pd.DataFrame, members: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.column_stack([within_month_rank(frame[m], frame["mes"]) for m in members])


@dataclass
class StackResult:
    weights: dict[int, np.ndarray]  # held-out month -> weights fitted on the other months
    scores: np.ndarray  # one score per row of ``frame``, from the weights that excluded its month
    gini_by_month: dict[int, float]


def lomo_stack(frame: pd.DataFrame, members: list[str] | tuple[str, ...], months=MONTHS, step: float = 0.1) -> StackResult:
    frame = frame.reset_index(drop=True)
    ranks = _rank_matrix(frame, members)
    y = frame["objetivo"].to_numpy()
    month_values = frame["mes"].to_numpy()
    weights: dict[int, np.ndarray] = {}
    scores = np.full(len(frame), np.nan)
    gini: dict[int, float] = {}
    for held_out in months:
        fit_mask = np.isin(month_values, [m for m in months if m != held_out])
        weights[held_out] = fit_rank_weights(ranks[fit_mask], y[fit_mask], month_values[fit_mask], step)
        test_mask = month_values == held_out
        scores[test_mask] = ranks[test_mask] @ weights[held_out]
        gini[held_out] = gini_score(y[test_mask], scores[test_mask])
    return StackResult(weights, scores, gini)


# --------------------------------------------------------------- best-of (LOMO)
@dataclass
class BestOfResult:
    chosen: dict[int, str]
    held_out_gini: dict[int, float]
    mean: float
    wins: dict[str, int]  # per-month point-estimate winners (no selection involved)


def lomo_best_of(table: pd.DataFrame) -> BestOfResult:
    """Pick the candidate with the best mean on the other months; score it on the held-out month.

    ``table``: rows = candidates (in priority order for ties), columns = months.
    """
    months = list(table.columns)
    chosen: dict[int, str] = {}
    held_out: dict[int, float] = {}
    for month in months:
        others = [m for m in months if m != month]
        means = np.round(table[others].mean(axis=1).to_numpy(dtype=float), 12)
        pick = table.index[int(np.argmax(means))]
        chosen[month] = str(pick)
        held_out[month] = float(table.loc[pick, month])
    wins = {str(name): 0 for name in table.index}
    for month in months:
        wins[str(table.index[int(np.argmax(np.round(table[month].to_numpy(dtype=float), 12)))])] += 1
    return BestOfResult(chosen, held_out, float(np.mean(list(held_out.values()))), wins)


# ------------------------------------------------------------------- bootstrap
def bootstrap_ginis(frame: pd.DataFrame, scores: dict[str, np.ndarray], months=MONTHS,
                    replicates: int = 2000, seed: int = 42) -> np.ndarray:
    """Gini draws, shape (replicates, models, months), with one shared client draw per replicate."""
    frame = frame.reset_index(drop=True)
    ids = pd.Index(frame["id_cliente"].unique())
    customer = ids.get_indexer(frame["id_cliente"])
    y = frame["objetivo"].to_numpy()
    masks = [frame["mes"].eq(m).to_numpy() for m in months]
    functions = [
        [_weighted_gini_preparation(y[mask], np.asarray(score)[mask], customer[mask]) for mask in masks]
        for score in scores.values()
    ]
    draws = np.empty((replicates, len(scores), len(months)))
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        for i, funcs in enumerate(functions):
            draws[r, i] = [func(counts) for func in funcs]
    if not np.isfinite(draws).all():
        raise ValueError("Bootstrap replicate lacks a class; cannot silently discard draws")
    return draws


def probability_of_best(draws: np.ndarray, names: list[str], month_index: int | None = None) -> dict[str, float]:
    """Share of replicates in which each model has the highest (mean over months, or one month) Gini."""
    values = draws.mean(axis=2) if month_index is None else draws[:, :, month_index]
    top = values.max(axis=1, keepdims=True)
    winners = np.isclose(values, top, rtol=0, atol=1e-12)
    share = (winners / winners.sum(axis=1, keepdims=True)).mean(axis=0)
    return {name: float(share[i]) for i, name in enumerate(names)}


def delta_table(draws: np.ndarray, names: list[str], points: np.ndarray, baseline: str) -> pd.DataFrame:
    """Paired deltas vs ``baseline``: point, 95% percentile CI (mean and per month), Holm p."""
    base = names.index(baseline)
    replicates = draws.shape[0]
    rows = []
    for i, name in enumerate(names):
        if i == base:
            continue
        delta = draws[:, i] - draws[:, base]
        rolling = delta.mean(axis=1)
        observed = points[i] - points[base]
        mean_delta = float(observed.mean())
        row = {"candidate": name, "delta_mean": mean_delta}
        row["ci_low_mean"], row["ci_high_mean"] = (float(v) for v in np.quantile(rolling, [0.025, 0.975]))
        for j, month in enumerate(MONTHS):
            low, high = np.quantile(delta[:, j], [0.025, 0.975])
            row.update({f"delta_{month}": float(observed[j]), f"ci_low_{month}": float(low), f"ci_high_{month}": float(high)})
        row["p_two_sided"] = float((1 + np.sum(np.abs(rolling - mean_delta) >= abs(mean_delta))) / (replicates + 1))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_two_sided"].tolist())
    return result


# -------------------------------------------------------------------- December
def _blend_december(members: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    frame = members.assign(mes=DECEMBER_MONTH)
    return blend_scores(frame, candidate)


def december_report(members: pd.DataFrame, candidates: list[Candidate], rolling_mean: dict[str, float],
                    current_submission: np.ndarray) -> dict:
    """Spearman between December blends of buildable candidates and with the current submission."""
    buildable = [c for c in candidates if c.members <= set(members.columns)]
    skipped = [c.name for c in candidates if c not in buildable]
    scores = {c.name: _blend_december(members, c) for c in buildable}
    ordered = sorted(scores, key=lambda n: -rolling_mean[n])
    top2 = ordered[:2]
    pairs = {f"{a} vs {b}": float(spearmanr(scores[a], scores[b]).statistic) for a, b in combinations(scores, 2)}
    vs_current = {n: float(spearmanr(scores[n], current_submission).statistic) for n in scores}
    return {"buildable": [c.name for c in buildable], "not_buildable": skipped, "top2_buildable": top2,
            "pairwise_spearman": pairs, "spearman_vs_current_submission": vs_current}


def align_december(member_predictions: pd.DataFrame, neural: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """December members in ``sample_submission`` order; assert every source covers exactly those ids."""
    order = sample["id_cliente"].to_numpy()
    checks = {
        "members_same_order_as_sample": bool(np.array_equal(member_predictions["id_cliente"].to_numpy(), order)),
        "neural_same_order_as_sample": bool(np.array_equal(neural["id_cliente"].to_numpy(), order)),
    }
    for label, frame in (("H4b2 members", member_predictions), ("neural", neural)):
        if frame["id_cliente"].duplicated().any() or set(frame["id_cliente"]) != set(order):
            raise ValueError(f"{label} do not cover exactly the sample_submission ids")
    index = pd.Index(order)
    out = pd.DataFrame({"id_cliente": order})
    for column, source in DECEMBER_MEMBERS.items():
        out[column] = member_predictions.set_index("id_cliente")[source].reindex(index).to_numpy()
    out["N1"] = neural.set_index("id_cliente")["N1"].reindex(index).to_numpy()
    return out, checks


# ------------------------------------------------------------------- the run
def _fmt(value: float) -> str:
    return f"{value:.5f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _report(run_id: str, summary: pd.DataFrame, deltas: pd.DataFrame, stack: StackResult, best_of: BestOfResult,
            wins_month: pd.DataFrame, p_best_month: pd.DataFrame, full_weights: np.ndarray, december: dict,
            reproduction: dict, replicates: int, step: float, h4b2_mean: float) -> str:
    lines = [f"# Ensemble decision `{run_id}`", "",
             "Rolling Sep/Oct/Nov out-of-fold predictions (train <=Aug->Sep, <=Sep->Oct, <=Oct->Nov). Candidate list fixed a priori; "
             f"every blend is a weighted mean of within-month percentile ranks. Paired client bootstrap: {replicates} resamples, one shared "
             "client draw across months and models; delta CIs are unadjusted percentile 95%; p is the two-sided centered bootstrap p with "
             "plus-one correction, Holm-adjusted across the family (fixed candidates + the two LOMO procedures).", "",
             "Reproduction checks (rank-averaging the stored members against the stored columns): "
             + ", ".join(f"{k} mean {v['recomputed']:.5f} vs stored {v['stored']:.5f}" for k, v in reproduction.items()), "",
             "## Candidates", "",
             "| Candidate | Sep | Oct | Nov | Mean | Std | dMean vs H4b2 [95% CI] | p Holm | P(best) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for _, row in summary.iterrows():
        cell = "-" if row["candidate"] == BASELINE else (
            f"{_signed(row['delta_mean'])} [{_signed(row['ci_low_mean'])}, {_signed(row['ci_high_mean'])}]")
        holm = "-" if row["candidate"] == BASELINE else f"{row['p_holm']:.3f}"
        pbest = "-" if pd.isna(row["p_best"]) else f"{row['p_best']:.3f}"
        lines.append(f"| {row['candidate']} | {_fmt(row['gini_202609'])} | {_fmt(row['gini_202610'])} | {_fmt(row['gini_202611'])} | "
                     f"{_fmt(row['mean'])} | {_fmt(row['std'])} | {cell} | {holm} | {pbest} |")
    lines += ["", f"`{STACK_NAME}` and `{BESTOF_NAME}` are out-of-selection estimates (see below); P(best) is only defined for the fixed candidates "
              "(share of bootstrap replicates in which the candidate has the highest mean Gini; ties split).", "",
              "## Per-month deltas vs H4b2 (95% CI)", "", "| Candidate | dSep | dOct | dNov |", "| --- | --- | --- | --- |"]
    for _, row in deltas.iterrows():
        cells = [f"{_signed(row[f'delta_{m}'])} [{_signed(row[f'ci_low_{m}'])}, {_signed(row[f'ci_high_{m}'])}]" for m in MONTHS]
        lines.append(f"| {row['candidate']} | " + " | ".join(cells) + " |")
    lines += ["", f"## Leave-one-month-out rank-weight stack (non-negative, sum 1, grid step {step})", "",
              f"Members: {', '.join(STACK_MEMBERS)}. Weights are fitted on two months (maximising their mean Gini) and the third is scored; "
              "the held-out month is never used for fitting.", "",
              "| Held-out month | " + " | ".join(STACK_MEMBERS) + " | Held-out Gini | H4b2 Gini |",
              "| --- | " + " | ".join("---:" for _ in STACK_MEMBERS) + " | ---: | ---: |"]
    h4b2_row = summary.loc[summary["candidate"] == BASELINE].iloc[0]
    for month in MONTHS:
        lines.append(f"| {month} | " + " | ".join(f"{w:.2f}" for w in stack.weights[month])
                     + f" | {_fmt(stack.gini_by_month[month])} | {_fmt(h4b2_row[f'gini_{month}'])} |")
    stack_mean = float(np.mean(list(stack.gini_by_month.values())))
    lines += ["", f"LOMO stack mean Gini {_fmt(stack_mean)} vs H4b2 {_fmt(h4b2_mean)} ({_signed(stack_mean - h4b2_mean)}). "
              "Descriptive in-sample weights fitted on all three months (not an estimate of anything): "
              + ", ".join(f"{m}={w:.2f}" for m, w in zip(STACK_MEMBERS, full_weights)) + ".", "",
              "## Best-of-list selection bias (LOMO)", "",
              "Choose the candidate with the best mean on two months, score it on the third.", "",
              "| Held-out month | Chosen on the other two | Held-out Gini | H4b2 Gini | Delta |", "| --- | --- | ---: | ---: | ---: |"]
    for month in MONTHS:
        gap = best_of.held_out_gini[month] - h4b2_row[f"gini_{month}"]
        lines.append(f"| {month} | {best_of.chosen[month]} | {_fmt(best_of.held_out_gini[month])} | "
                     f"{_fmt(h4b2_row[f'gini_{month}'])} | {_signed(gap)} |")
    lines += ["", f"Best-of LOMO mean Gini {_fmt(best_of.mean)} vs H4b2 {_fmt(h4b2_mean)} ({_signed(best_of.mean - h4b2_mean)}).", "",
              "## How often each candidate is best", "",
              "Point-estimate wins per month, and share of bootstrap replicates in which the candidate is best (per month and on the mean).", "",
              "| Candidate | Wins (of 3 months) | P(best) Sep | P(best) Oct | P(best) Nov | P(best) mean |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in wins_month.index:
        lines.append(f"| {name} | {int(wins_month.loc[name, 'wins'])} | " + " | ".join(f"{p_best_month.loc[name, m]:.3f}" for m in (*MONTHS, "mean")) + " |")
    lines += ["", "## December sanity", ""]
    lines += [f"- Buildable from existing December outputs (members B0, H1b, N1): {', '.join(december['buildable'])}.",
              f"- Not buildable without a new December fit (need cat+F1 or xgb): {', '.join(december['not_buildable']) or 'none'}.",
              f"- Top 2 buildable by rolling mean: {', '.join(december['top2_buildable'])}.",
              f"- Id order checks: {december['id_checks']}.", "", "| Pair | Spearman |", "| --- | ---: |"]
    lines += [f"| {k} | {v:.4f} |" for k, v in december["pairwise_spearman"].items()]
    lines += ["", "| Candidate | Spearman vs current H4b2 submission |", "| --- | ---: |"]
    lines += [f"| {k} | {v:.4f} |" for k, v in december["spearman_vs_current_submission"].items()]
    lines += ["", "No submission file was written by this run."]
    return "\n".join(lines) + "\n"


def run_ensemble_search(
    data_dir: str | Path = "data",
    experiment_root: str | Path = "experiments/ensembles",
    round1_oof: str | Path = "experiments/hypotheses/20261006T100749205414Z_a714ebcd/oof_predictions.csv",
    round2_oof: str | Path = "experiments/hypotheses/20261006T143722753859Z_a714ebcd_r2/oof_predictions.csv",
    neural_dir: str | Path = "experiments/neural",
    h4b2_members: str | Path = "experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/member_predictions.csv",
    current_submission: str | Path = "experiments/final/20261006T101910790713Z_a714ebcd_ensemble-h4b2/submission.csv",
    replicates: int = 2000,
    seed: int = 42,
    step: float = 0.1,
) -> dict:
    data_dir, experiment_root, neural_dir = Path(data_dir), Path(experiment_root), Path(neural_dir)
    train_path, sample_path = data_dir / "train.csv", data_dir / "sample_submission.csv"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{sha256_file(train_path)[:8]}_ensemble-search"
    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Self-contained copies of the out-of-tree neural outputs (small CSVs, tracked).
    neural_copy = run_dir / "neural"
    neural_copy.mkdir()
    for name in ("oof_neural.csv", "test_neural.csv", "results.md"):
        shutil.copy2(neural_dir / name, neural_copy / name)

    r1 = pd.read_csv(round1_oof)
    r2 = pd.read_csv(round2_oof)
    neural = pd.read_csv(neural_copy / "oof_neural.csv")
    keep = [*KEYS, "objetivo"]
    joined = join_oof([
        r2[[*keep, *ROUND2_MEMBERS.values(), "H4b2_rank_lgbm_catboost", "E1_rank_inner_selected"]]
        .rename(columns={v: k for k, v in ROUND2_MEMBERS.items()}),
        neural[[*keep, "N1"]],
    ])
    # Round 1 must carry the same B0/H1b predictions as round 2 (same rows, same values).
    r1_check = join_oof([joined[[*keep, "B0", "H1b"]], r1[[*keep, "B0_lgbm_D", "H1b_catboost"]]])
    if not (np.allclose(r1_check["B0"], r1_check["B0_lgbm_D"]) and np.allclose(r1_check["H1b"], r1_check["H1b_catboost"])):
        raise ValueError("round-1 and round-2 OOF disagree on B0/H1b predictions")
    if set(joined["mes"].unique()) != set(MONTHS):
        raise ValueError(f"unexpected months {sorted(joined['mes'].unique())}")
    joined_path = run_dir / "joined_oof.csv"
    joined.to_csv(joined_path, index=False)

    candidates = fixed_candidates()
    names = [c.name for c in candidates]
    scores = {c.name: blend_scores(joined, c) for c in candidates}
    points = {n: monthly_ginis(joined, s) for n, s in scores.items()}

    reproduction = {}
    for label, candidate_name, stored_col in (("H4b2", "H4b2", "H4b2_rank_lgbm_catboost"), ("E1", "E1", "E1_rank_inner_selected")):
        stored = float(monthly_ginis(joined, joined[stored_col].to_numpy()).mean())
        recomputed = float(points[candidate_name].mean())
        if abs(stored - recomputed) > 1e-4:
            raise ValueError(f"{label} recomputed mean {recomputed} differs from stored {stored}")
        reproduction[label] = {"stored": stored, "recomputed": recomputed}

    stack = lomo_stack(joined, STACK_MEMBERS, MONTHS, step)
    full_weights = fit_rank_weights(_rank_matrix(joined, STACK_MEMBERS), joined["objetivo"].to_numpy(), joined["mes"].to_numpy(), step)
    table = pd.DataFrame({m: [points[n][j] for n in names] for j, m in enumerate(MONTHS)}, index=names)
    best_of = lomo_best_of(table)
    bestof_scores = np.empty(len(joined))
    for month, chosen in best_of.chosen.items():
        mask = joined["mes"].eq(month).to_numpy()
        bestof_scores[mask] = scores[chosen][mask]
    procedures = {STACK_NAME: stack.scores, BESTOF_NAME: bestof_scores}
    all_scores = {**scores, **procedures}
    all_points = {**points, STACK_NAME: monthly_ginis(joined, stack.scores), BESTOF_NAME: monthly_ginis(joined, bestof_scores)}
    all_names = list(all_scores)

    draws = bootstrap_ginis(joined, all_scores, MONTHS, replicates, seed)
    deltas = delta_table(draws, all_names, np.array([all_points[n] for n in all_names]), BASELINE)
    n_fixed = len(names)
    fixed_draws = draws[:, :n_fixed]
    p_mean = probability_of_best(fixed_draws, names)
    p_months = {m: probability_of_best(fixed_draws, names, j) for j, m in enumerate(MONTHS)}
    p_best_month = pd.DataFrame({**{m: pd.Series(p_months[m]) for m in MONTHS}, "mean": pd.Series(p_mean)})
    wins_month = pd.DataFrame({"wins": pd.Series(best_of.wins)})

    rows = []
    for name in all_names:
        values = all_points[name]
        row = {"candidate": name, **{f"gini_{m}": float(values[j]) for j, m in enumerate(MONTHS)},
               "mean": float(values.mean()), "std": float(values.std()), "p_best": p_mean.get(name, np.nan),
               "wins_months": best_of.wins.get(name, np.nan)}
        rows.append(row)
    summary = pd.DataFrame(rows).merge(deltas, on="candidate", how="left")
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    members_dec, id_checks = align_december(
        pd.read_csv(h4b2_members), pd.read_csv(neural_copy / "test_neural.csv"), pd.read_csv(sample_path))
    december = december_report(members_dec, candidates, {n: float(points[n].mean()) for n in names},
                               pd.read_csv(current_submission).set_index("id_cliente")["prediccion"].reindex(members_dec["id_cliente"]).to_numpy())
    december["id_checks"] = id_checks

    h4b2_mean = float(points[BASELINE].mean())
    weights_path = run_dir / "stack_weights.json"
    write_json(weights_path, {"members": list(STACK_MEMBERS), "grid_step": step,
                              "lomo_weights": {str(m): w.tolist() for m, w in stack.weights.items()},
                              "lomo_gini": {str(m): g for m, g in stack.gini_by_month.items()},
                              "in_sample_all_months": full_weights.tolist(), "best_of_lomo": best_of.chosen})
    december_path = run_dir / "december_spearman.json"
    write_json(december_path, december)
    results_path = run_dir / "results.md"
    results_path.write_text(_report(run_id, summary, deltas, stack, best_of, wins_month, p_best_month, full_weights, december,
                                    reproduction, replicates, step, h4b2_mean), encoding="utf-8")

    outputs = [results_path, summary_path, joined_path, weights_path, december_path]
    inputs = [train_path, sample_path, round1_oof, round2_oof, h4b2_members, current_submission,
              neural_copy / "oof_neural.csv", neural_copy / "test_neural.csv", neural_copy / "results.md"]
    manifest = build_run_manifest(
        run_id, "ensemble-search", "rank-blend-candidates", input_paths=inputs, model_path=weights_path,
        output_paths=[p for p in outputs if p != weights_path],
        parameters={"candidates": {c.name: [[w, list(m)] for w, m in c.groups] for c in candidates},
                    "stack_members": list(STACK_MEMBERS), "grid_step": step, "replicates": replicates, "seed": seed,
                    "baseline": BASELINE, "holm_family": "all non-baseline rows (fixed candidates + LOMO procedures)",
                    "months": list(MONTHS)},
        metrics={"mean_gini": {n: float(all_points[n].mean()) for n in all_names},
                 "december": {"top2_buildable": december["top2_buildable"]}},
        code_paths=[Path("src/datafest") / n for n in ("ensemble_search.py", "hypotheses.py", "diagnostics.py", "metrics.py", "lineage.py")]
        + [Path("pyproject.toml"), Path("uv.lock")],
        environment={"python": platform.python_version(), "datafest": __version__},
    )
    write_json(run_dir / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": str(run_dir), "summary": str(summary_path), "results": str(results_path)}
