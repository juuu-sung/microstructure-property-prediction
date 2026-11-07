"""User-run learning curve: fixed DINO features plus subset-selected Ridge.

Train sizes 100/200/300/400, three nested random subsets, fixed outer validation.
No images, torch, test data, downloads, final all-data fitting or submission.
--validate-only validates inputs and creates memberships in memory, with no fit/output.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

if __package__:
    from . import evaluate_frozen_resnet as core
    from . import evaluate_frozen_dinov2 as dino
    from . import evaluate_feature_fusion as references_module
else:
    import evaluate_frozen_resnet as core
    import evaluate_frozen_dinov2 as dino
    import evaluate_feature_fusion as references_module

np = core.np
ROOT = core.ROOT
FEATURE_NAME = "cls_patch_mean_std"
ALPHAS = (10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0)
TRAIN_SIZES = (100, 200, 300, 400)
REPEATS = 3
ANCHOR_PREDICTION_ATOL = 1e-7
ANCHOR_SCORE_ATOL = 1e-8
SAMPLING_SALT = 1001


def load_inputs(features_path, train_path, folds_path, baseline_path, audit_path,
                previous_frozen_path, previous_dino_path, seed, *, metadata_path=None,
                previous_dino_summary=None, expected_n=500):
    data, frozen_refs = dino.load_inputs(features_path, train_path, folds_path, baseline_path,
        audit_path, previous_frozen_path, seed, metadata_path=metadata_path, expected_n=expected_n)
    summary_path = previous_dino_summary or previous_dino_path.with_name("summary.json")
    references = references_module.load_previous_dino(data, frozen_refs, previous_dino_path, summary_path, seed)
    rows = core.align_rows(core.read_rows(previous_dino_path,
        {"ID", "hardness", "fold", FEATURE_NAME + "_nested"}), data.ids, previous_dino_path)
    anchor_predictions = core.finite_column(rows, FEATURE_NAME + "_nested", previous_dino_path)
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    if old.get("alpha_candidates") != list(ALPHAS) or old.get("inner_splits") != 3:
        raise ValueError("Previous DINO anchor must use the fixed six alpha candidates and three inner folds")
    reported = old.get("diagnostics", {}).get(FEATURE_NAME + "_nested", {}).get("overall", {}).get("rmse")
    if not isinstance(reported, (float, int)) or abs(core.metrics(data.y, anchor_predictions)["rmse"] - reported) > ANCHOR_SCORE_ATOL:
        raise ValueError("Previous per-representation DINO anchor OOF/summary RMSE mismatch")
    anchor_records = old.get("selections", [])
    if len(anchor_records) != 5 or {r.get("outer_fold") for r in anchor_records} != set(range(5)):
        raise ValueError("Previous DINO anchor needs exactly five outer selections")
    selections = {}
    for record in anchor_records:
        fold = record["outer_fold"]
        train_n = int(np.sum(data.folds != fold))
        if (record.get("inner_seed") != seed + fold or record.get("outer_train_n") != train_n or
                record.get("outer_valid_n") != len(data.y) - train_n):
            raise ValueError("Previous DINO anchor fold sizes or inner seed do not match")
        candidates = [copy.deepcopy(c) for c in record.get("candidates", []) if c.get("representation") == FEATURE_NAME]
        if len(candidates) != len(ALPHAS) or {c.get("alpha") for c in candidates} != set(ALPHAS):
            raise ValueError("Previous DINO anchor per-representation candidate grid differs")
        sizes = record.get("inner_validation_sizes", [])
        if len(sizes) != 3 or sum(sizes) != train_n or min(sizes) < 1:
            raise ValueError("Previous DINO anchor inner validation sizes are invalid")
        for candidate in candidates:
            scores = candidate.get("inner_fold_mse", [])
            if len(scores) != 3 or not np.isfinite([candidate.get("inner_mse"), *scores]).all():
                raise ValueError("Previous DINO anchor has invalid inner scores")
            if abs(float(np.average(scores, weights=sizes)) - candidate["inner_mse"]) > ANCHOR_SCORE_ATOL:
                raise ValueError("Previous DINO anchor pooled inner MSE is inconsistent")
        winner = min(candidates, key=lambda c: (c["inner_mse"], -c["alpha"]))
        if winner != record.get("per_representation", {}).get(FEATURE_NAME):
            raise ValueError("Previous DINO anchor alpha selection is inconsistent")
        selections[fold] = {"candidates": candidates, "winner": winner, "inner_validation_sizes": sizes}
    values = data.features[FEATURE_NAME]
    values.flags.writeable = False
    data.features = {FEATURE_NAME: values}
    data.provenance["learning_curve_source"] = {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__))}
    data.provenance["shared_sources"] = {name: {"path": str(Path(module.__file__).resolve()), "sha256": core.sha256(Path(module.__file__))}
        for name, module in {"nested_core": core, "dino_validator": dino, "reference_validator": references_module}.items()}
    return data, references, {"predictions": anchor_predictions, "selections": selections,
        "source_oof": str(previous_dino_path.resolve()), "prediction_column": FEATURE_NAME + "_nested"}


def build_memberships(folds, seed, *, train_sizes=TRAIN_SIZES, repeats=REPEATS):
    """Labels are deliberately absent: subset selection uses indices only."""
    seed = core.validate_seed(seed)
    sizes = tuple(train_sizes)
    if (not sizes or sizes != tuple(sorted(set(sizes))) or min(sizes) < 1 or repeats < 1 or
            set(folds) != set(range(5))):
        raise ValueError("Invalid membership sizes, repeats, or outer folds")
    memberships = {}
    for fold in range(5):
        outer_train = np.flatnonzero(folds != fold)
        if sizes[-1] != len(outer_train):
            raise ValueError("Largest training size must equal every outer training set")
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, fold, repeat, SAMPLING_SALT]))
            permutation = rng.permutation(outer_train)
            for size in sizes:
                memberships[(fold, repeat, size)] = np.sort(permutation[:size])
    return memberships


def planned_fit_count(train_sizes=TRAIN_SIZES, repeats=REPEATS, alphas=ALPHAS):
    return 5 * (repeats * (len(train_sizes) - 1) + 1) * (3 * len(alphas) + 1)


def fit_subset(data, train_indices, validation_indices, inner_seed, *, alphas=ALPHAS):
    """Only this subset's labels enter inner stratification, scaling and fitting."""
    from sklearn.model_selection import StratifiedKFold
    X = data.features[FEATURE_NAME]
    train_x, train_y = X[train_indices].astype(np.float64), data.y[train_indices]
    if len(train_y) < 30:
        raise ValueError("Each training subset needs at least 30 samples for inner target deciles")
    strata = core.rank_groups(train_y, 10)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=inner_seed).split(train_y, strata))
    candidates = []
    for alpha in alphas:
        inner = np.full(len(train_y), np.nan)
        scores = []
        for inner_train, inner_valid in splits:
            estimator = core.make_estimator(alpha)
            estimator.fit(train_x[inner_train], train_y[inner_train])
            inner[inner_valid] = estimator.predict(train_x[inner_valid])
            scores.append(float(np.mean((inner[inner_valid] - train_y[inner_valid]) ** 2)))
        if not np.isfinite(inner).all():
            raise ValueError("Inner predictions are incomplete/nonfinite")
        candidates.append({"representation": FEATURE_NAME, "alpha": alpha,
            "inner_mse": float(np.mean((inner - train_y) ** 2)), "inner_fold_mse": scores})
    winner = min(candidates, key=lambda candidate: (candidate["inner_mse"], -candidate["alpha"]))
    estimator = core.make_estimator(winner["alpha"])
    estimator.fit(train_x, train_y)
    train_pred = estimator.predict(train_x)
    valid_pred = estimator.predict(X[validation_indices].astype(np.float64))
    if not np.isfinite(train_pred).all() or not np.isfinite(valid_pred).all():
        raise ValueError("Training or outer predictions contain nonfinite values")
    return {"candidates": candidates, "winner": dict(winner), "inner_seed": inner_seed,
        "inner_validation_sizes": [len(valid) for _, valid in splits],
        "train_metrics": core.metrics(train_y, train_pred),
        "validation_metrics": core.metrics(data.y[validation_indices], valid_pred),
        "validation_indices": validation_indices, "validation_predictions": valid_pred,
        "subset_target_mean": float(train_y.mean()), "subset_target_std": float(train_y.std()),
        "subset_target_min": float(train_y.min()), "subset_target_max": float(train_y.max())}


def validate_anchor(full_records, data, anchor, *, prediction_atol=ANCHOR_PREDICTION_ATOL, score_atol=ANCHOR_SCORE_ATOL):
    prediction = np.full(len(data.y), np.nan)
    max_score_delta = 0.0
    for fold in range(5):
        record, old = full_records[fold], anchor["selections"][fold]
        if record["winner"]["alpha"] != old["winner"]["alpha"]:
            raise ValueError(f"400-anchor selected alpha mismatch in fold {fold}; no results saved")
        if record["inner_validation_sizes"] != old["inner_validation_sizes"]:
            raise ValueError(f"400-anchor inner fold-size mismatch in fold {fold}; no results saved")
        old_candidates = {c["alpha"]: c for c in old["candidates"]}
        if {c["alpha"] for c in record["candidates"]} != set(old_candidates):
            raise ValueError("400-anchor alpha grid mismatch; no results saved")
        for candidate in record["candidates"]:
            previous = old_candidates[candidate["alpha"]]
            delta = float(np.max(np.abs(np.array([candidate["inner_mse"], *candidate["inner_fold_mse"]]) -
                                            np.array([previous["inner_mse"], *previous["inner_fold_mse"]]))))
            max_score_delta = max(max_score_delta, delta)
        prediction[record["validation_indices"]] = record["validation_predictions"]
    max_prediction_delta = float(np.max(np.abs(prediction - anchor["predictions"])))
    if not np.isfinite(prediction).all() or max_prediction_delta > prediction_atol or max_score_delta > score_atol:
        raise ValueError(f"400-anchor reproducibility failed: max prediction difference={max_prediction_delta:g}, "
                         f"max inner MSE difference={max_score_delta:g}; no results saved")
    return {"status": "matched", "max_absolute_prediction_difference": max_prediction_delta,
        "max_absolute_inner_mse_difference": max_score_delta, "prediction_atol": prediction_atol,
        "inner_score_atol": score_atol, "selected_alphas_exactly_match": True,
        "reference_prediction_column": anchor["prediction_column"]}


def evaluate_curve(data, memberships, seed, threads=2, *, anchor=None,
                   train_sizes=TRAIN_SIZES, repeats=REPEATS, alphas=ALPHAS):
    from threadpoolctl import threadpool_limits
    alphas = core.validate_alphas(alphas)
    largest = train_sizes[-1]
    records, full = [], {}
    with threadpool_limits(limits=threads):
        # Check reproducibility early. Reference predictions never enter fitting
        # or selection; comparison can only halt this run on incompatibility.
        for fold in range(5):
            full[fold] = fit_subset(data, memberships[(fold, 0, largest)], np.flatnonzero(data.folds == fold),
                                    seed + fold, alphas=alphas)
        anchor_check = validate_anchor(full, data, anchor) if anchor is not None else {"status": "not_requested_for_synthetic_test"}
        print(f"Full-size anchor: {anchor_check['status']}", flush=True)
        for fold in range(5):
            validation = np.flatnonzero(data.folds == fold)
            for repeat in range(repeats):
                for size in train_sizes:
                    value = full[fold] if size == largest else fit_subset(data, memberships[(fold, repeat, size)],
                        validation, seed + fold, alphas=alphas)
                    records.append({**value, "outer_fold": fold, "repeat": repeat, "train_size": size,
                        "full_size_reused": size == largest and repeat > 0,
                        "fit_instance": f"fold{fold}_full" if size == largest else f"fold{fold}_repeat{repeat}_n{size}"})
            print(f"Outer fold {fold}: completed sizes {list(train_sizes)}, repeats={repeats}; full-size fitted once", flush=True)
    return records, anchor_check


def summarize(data, records, *, train_sizes=TRAIN_SIZES, repeats=REPEATS):
    curves, oof_rows, fold_rows, search_rows = [], [], [], []
    target_bins, noise_bins = core.rank_groups(data.y, 10), core.rank_groups(data.noise, 4)
    for repeat in range(repeats):
        for size in train_sizes:
            group = [r for r in records if r["repeat"] == repeat and r["train_size"] == size]
            if len(group) != 5 or {r["outer_fold"] for r in group} != set(range(5)):
                raise ValueError("Curve records must cover each of the five outer folds exactly once")
            pred = np.full(len(data.y), np.nan)
            train_sse, train_count = 0.0, 0
            for r in group:
                pred[r["validation_indices"]] = r["validation_predictions"]
                train_sse += r["train_metrics"]["sse"]
                train_count += r["train_metrics"]["n"]
                fold_rows.append({"repeat": repeat, "train_size": size, "outer_fold": r["outer_fold"],
                    "selected_alpha": r["winner"]["alpha"], "inner_mse": r["winner"]["inner_mse"],
                    "train_rmse": r["train_metrics"]["rmse"], "validation_rmse": r["validation_metrics"]["rmse"],
                    "train_sse": r["train_metrics"]["sse"], "validation_sse": r["validation_metrics"]["sse"],
                    **{name: r[name] for name in ("subset_target_mean", "subset_target_std", "subset_target_min", "subset_target_max", "full_size_reused", "fit_instance")}})
                # The identical full-size search is recorded once, not counted
                # as three independent searches.
                if not r["full_size_reused"]:
                    for c in r["candidates"]:
                        search_rows.append({"outer_fold": r["outer_fold"], "repeat": repeat, "train_size": size,
                            "fit_instance": r["fit_instance"], "inner_seed": r["inner_seed"], "alpha": c["alpha"],
                            "inner_mse": c["inner_mse"], **{f"inner_fold{j}_mse": x for j, x in enumerate(c["inner_fold_mse"])},
                            "selected": c == r["winner"]})
            if not np.isfinite(pred).all():
                raise ValueError("Incomplete/nonfinite learning-curve OOF")
            row = {"repeat": repeat, "train_size": size, "oof_rmse": core.metrics(data.y, pred)["rmse"],
                "train_resubstitution_rmse": float(np.sqrt(train_sse / train_count)),
                "train_prediction_count_with_cross_fold_overlap": train_count,
                "full_size_reused": size == train_sizes[-1] and repeat > 0}
            for name, mask in {"tails20": (target_bins == 0) | (target_bins == 9),
                               "middle80": (target_bins > 0) & (target_bins < 9),
                               **{f"noise_q{j+1}": noise_bins == j for j in range(4)}}.items():
                row[name + "_rmse"] = core.metrics(data.y[mask], pred[mask])["rmse"]
            curves.append(row)
            oof_rows.extend({"repeat": repeat, "train_size": size, "ID": str(ident), "fold": int(data.folds[i]),
                "hardness": float(data.y[i]), "prediction": float(pred[i]), "error": float(pred[i]-data.y[i]),
                "full_size_reused": row["full_size_reused"]} for i, ident in enumerate(data.ids))
    aggregate = []
    for size in train_sizes:
        group = [r for r in curves if r["train_size"] == size]
        item = {"train_size": size, "repeat_count": repeats, "independently_fitted_subset_repeats": 1 if size == train_sizes[-1] else repeats}
        for metric in ("oof_rmse", "train_resubstitution_rmse", "tails20_rmse", "middle80_rmse", "noise_q1_rmse", "noise_q2_rmse", "noise_q3_rmse", "noise_q4_rmse"):
            values = np.array([r[metric] for r in group])
            for label, value in {"mean": values.mean(), "std": values.std(ddof=1) if repeats > 1 and size != train_sizes[-1] else 0.0,
                                 "min": values.min(), "max": values.max()}.items():
                item[metric + "_" + label] = float(value)
        aggregate.append(item)
    before, after = train_sizes[-2:]
    paired = [{"repeat": repeat, "from_train_size": before, "to_train_size": after,
        "delta_oof_rmse": next(r["oof_rmse"] for r in curves if r["repeat"] == repeat and r["train_size"] == after) -
                          next(r["oof_rmse"] for r in curves if r["repeat"] == repeat and r["train_size"] == before)} for repeat in range(repeats)]
    return {"curve_by_repeat": curves, "curve_summary": aggregate, "paired_300_to_400": paired,
            "fold_metrics": fold_rows, "inner_search": search_rows, "oof": oof_rows}


def save_plot(output_dir, tables):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "skipped", "reason": "matplotlib unavailable; numeric reports remain complete; no installation attempted"}
    rows = tables["curve_summary"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, label in (("oof_rmse", "Outer validation"), ("train_resubstitution_rmse", "Training (resubstitution)")):
        ax.errorbar([r["train_size"] for r in rows], [r[metric + "_mean"] for r in rows],
                    yerr=[r[metric + "_std"] for r in rows], marker="o", capsize=4, label=label)
    ax.set(xlabel="Training samples per outer fold", ylabel="RMSE (hardness)",
           title="DINO Ridge learning curve: descriptive repeat variation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.01, "Largest subset is fitted once and reused; zero spread is by design. Error bars are not confidence intervals.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "saved", "path": str(path.resolve())}


def build_markdown(summary):
    size_text = "·".join(str(size) for size in summary["train_sizes"])
    previous_size, full_size = summary["train_sizes"][-2:]
    lines = ["# DINO Ridge 학습곡선", "",
        f"기존 5-fold 검증 {summary['n_train']//5}장은 고정하고 각 outer 학습 {full_size}장 중 {size_text}장을 사용했습니다. 각 크기에서 alpha를 inner 3-fold로 다시 선택했습니다.", "",
        "| 학습 장수/fold | OOF RMSE 평균 | 반복 표준편차 | 최소–최대 | 학습 RMSE 평균 |", "|---|---:|---:|---:|---:|"]
    for r in summary["curve_summary"]:
        lines.append(f"| {r['train_size']} | {r['oof_rmse_mean']:.6f} | {r['oof_rmse_std']:.6f} | {r['oof_rmse_min']:.6f}–{r['oof_rmse_max']:.6f} | {r['train_resubstitution_rmse_mean']:.6f} |")
    lines += ["", "반복마다 500장 전체 OOF RMSE를 먼저 계산한 후 세 반복의 평균·표준편차·범위를 구했습니다. 반복 예측을 평균한 앙상블 점수가 아닙니다.", "",
        f"| 반복 | {previous_size}→{full_size} OOF RMSE 변화 |", "|---|---:|"]
    for r in summary["paired_300_to_400"]:
        lines.append(f"| {r['repeat']} | {r['delta_oof_rmse']:+.6f} |")
    lines += ["", "음수면 학습 표본이 늘었을 때 개선입니다. 400장은 한 번 적합한 결과를 세 반복에서 재사용하므로 표준편차 0은 설계상 결과이며 불확실성 0을 뜻하지 않습니다.", "",
        f"기존 동일 DINO 표현의 400장 OOF 재현 검사: {summary['anchor_check']['status']}. 최대 예측 차이 {summary['anchor_check']['max_absolute_prediction_difference']:.3g}, 최대 inner MSE 차이 {summary['anchor_check']['max_absolute_inner_mse_difference']:.3g}.", "",
        "## 해석 범위", "",
        "- 부분집합은 outer train 인덱스의 고정 무작위 순열에서 앞부분을 선택합니다. 선정 과정은 y를 전혀 보지 않으며 100⊂200⊂300⊂400의 포함관계를 가집니다. 저장 순서는 원자료 인덱스 순입니다.",
        "- inner target decile은 해당 부분집합의 y로만 계산합니다. X/y StandardScaler도 매 inner 학습 부분에서 다시 적합합니다.",
        "- alpha 후보는 10, 100, 1000, 3000, 10000, 30000이며 원단위 pooled inner MSE로 선택합니다. 동점은 큰 alpha를 우선합니다.",
        "- 학습 RMSE는 선택한 모델의 학습 표본 재대입 값으로 낙관적입니다. 이를 outer OOF 일반화 성능과 구분합니다.",
        "- 반복과 outer fold는 데이터를 공유합니다. 표준편차는 부분집합 구성에 따른 기술통계이며 독립 표본 신뢰구간이 아닙니다.",
        "- 400장까지의 내부 학습곡선이므로 새 데이터 500장 이상에서의 개선을 보장하지 않습니다. 같은 500장을 반복한 연구 선택 편향도 남습니다.",
        f"- 기존 25/25/50 조합 OOF RMSE {summary['reference_context']['current_blend_rmse']:.6f}는 참고 맥락입니다. 이 곡선은 동일 조합이 아니라 DINO Ridge 한 모델입니다.",
        f"- 실제 적합 횟수: {summary['estimator_fit_count']}회/seed. 400장 fit을 반복마다 중복 실행하지 않습니다. 최종 전체 500장 적합, test, 이미지, 신경망 forward, 다운로드는 없습니다.",
        "- 타깃 분포 통계와 꼬리/잡음 진단은 부분집합 선정 이후 계산되며 표본 선택에 사용하지 않았습니다.",
        f"- 그래프: {summary['plot']['status']}. 숫자 결과는 CSV와 summary.json을 기준으로 확인하세요.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, data, references, memberships, records, anchor_check, tables, seed, threads, elapsed,
                  *, train_sizes=TRAIN_SIZES, repeats=REPEATS, alphas=ALPHAS):
    import scipy
    import sklearn
    membership_rows = [{"outer_fold": fold, "repeat": repeat, "train_size": size, "ID": str(data.ids[index]),
                        "source_row_index": int(index)} for (fold, repeat, size), indices in memberships.items() for index in indices]
    # No optional plotting library is installed automatically.
    try:
        plot = save_plot(output_dir, tables)
    except Exception as exc:
        plot = {"status": "skipped", "reason": f"Optional plot failed ({type(exc).__name__}: {exc}); numeric reports remain complete"}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "fixed_dino_ridge_learning_curve",
        "seed": seed, "threads": threads, "elapsed_seconds": elapsed, "n_train": len(data.y),
        "source_feature": FEATURE_NAME, "source_feature_dimension": data.features[FEATURE_NAME].shape[1],
        "train_sizes": list(train_sizes), "subset_repeats": repeats, "alpha_candidates": list(alphas),
        "outer_splits": 5, "inner_splits": 3, "estimator_fit_count": planned_fit_count(train_sizes, repeats, alphas),
        "sampling": {"label_independent": True, "algorithm": "default_rng(SeedSequence([seed, outer_fold, repeat, 1001])).permutation(outer_train_indices); prefix; sort original row indices",
            "target_statistics_used_for_selection": False, "nested_subsets": True},
        "primary_interpretation": "Four descriptive per-repeat pooled OOF RMSE values; not an ensemble of repeat predictions and not a model-selection score.",
        "full_size_reused_across_repeats": True, "repeat_std_is_confidence_interval": False,
        "anchor_check": anchor_check, "input_files": data.provenance, "feature_metadata": data.metadata,
        "reference_context": {"current_blend_rmse": core.metrics(data.y, references[references_module.CURRENT_REFERENCE])["rmse"],
            "note": "The 25/25/50 ensemble is context only; the curve evaluates one DINO Ridge representation."},
        "curve_by_repeat": tables["curve_by_repeat"], "curve_summary": tables["curve_summary"],
        "paired_300_to_400": tables["paired_300_to_400"], "plot": plot,
        "selections": [{k: v for k, v in r.items() if k not in ("validation_indices", "validation_predictions")} for r in records],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "scope": "Cached train features only; no images/test, no neural forward/download, no all-data final fit/submission."}
    outputs = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        "report.md": build_markdown(summary), "memberships.csv": core.csv_text(membership_rows),
        **{name + ".csv": core.csv_text(rows) for name, rows in tables.items()}}
    for name, content in outputs.items():
        path = output_dir / name
        temporary = output_dir / (name + f".{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, path)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--features", type=Path, default=ROOT / "reports/frozen_dinov2_small/features.npz")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "open/train.csv")
    parser.add_argument("--audit-features", type=Path, default=ROOT / "reports/audit_train_features.csv")
    for flag in ("folds", "baseline", "previous-frozen-oof", "previous-dino-oof", "previous-dino-summary", "output-dir"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    defaults = {"folds": ROOT / ("reports/folds.csv" if args.seed == 42 else f"reports/benchmark_folds_seed{args.seed}.csv"),
        "baseline": ROOT / f"reports/benchmark_oof_seed{args.seed}.csv",
        "previous_frozen_oof": ROOT / f"reports/frozen_resnet18/seed{args.seed}/oof.csv",
        "previous_dino_oof": ROOT / f"reports/frozen_dinov2_small/seed{args.seed}/oof.csv",
        "output_dir": ROOT / f"reports/dinov2_learning_curve/seed{args.seed}"}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.previous_dino_summary is None:
        args.previous_dino_summary = args.previous_dino_oof.with_name("summary.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    seed = core.validate_seed(args.seed)
    if np is None or args.threads < 1:
        raise ValueError("NumPy and positive --threads are required")
    if not args.validate_only:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError("--output-dir must be a directory")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise ValueError("Output directory contains results; use a new directory or explicit --overwrite")
        protected = [args.features, args.train_csv, args.folds, args.baseline, args.audit_features,
                     args.previous_frozen_oof, args.previous_dino_oof, args.previous_dino_summary,
                     args.metadata or args.features.with_name("metadata.json")]
        if any(path.resolve().parent == args.output_dir.resolve() for path in protected):
            raise ValueError("Output directory contains a protected input/reference")
    started = time.perf_counter()
    data, references, anchor = load_inputs(args.features, args.train_csv, args.folds, args.baseline, args.audit_features,
        args.previous_frozen_oof, args.previous_dino_oof, seed, metadata_path=args.metadata,
        previous_dino_summary=args.previous_dino_summary)
    memberships = build_memberships(data.folds, seed)
    if args.validate_only:
        print(json.dumps({"status": "validated_inputs_and_memberships", "seed": seed, "n_train": len(data.y),
            "train_sizes": TRAIN_SIZES, "repeats": REPEATS, "membership_sets": len(memberships),
            "sampling_uses_labels": False, "planned_estimator_fits": planned_fit_count(), "estimators_fitted": 0,
            "anchor_status": "reference schema/hashes verified; numerical reproduction requires user-run fitting",
            "output_written": False, "images_or_test_read": False}, indent=2))
        return 0
    records, anchor_check = evaluate_curve(data, memberships, seed, args.threads, anchor=anchor)
    tables = summarize(data, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_outputs(args.output_dir, data, references, memberships, records, anchor_check, tables, seed, args.threads, time.perf_counter()-started)
    for row in tables["curve_summary"]:
        print(f"train_size={row['train_size']}: OOF RMSE mean={row['oof_rmse_mean']:.6f}, std={row['oof_rmse_std']:.6f}; train RMSE={row['train_resubstitution_rmse_mean']:.6f}")
    if summary["plot"]["status"] == "skipped":
        print(f"Plot skipped: {summary['plot']['reason']}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Learning-curve evaluation stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
