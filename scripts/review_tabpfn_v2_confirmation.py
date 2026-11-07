"""Audit saved confirmation results only; no fitting, inference or test reads."""
from __future__ import annotations

import csv
import io
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

import evaluate_tabpfn_v2_confirmation as c
import review_tabpfn_v2 as audit

t = c.t


def csv_strings(rows):
    return list(csv.DictReader(io.StringIO(t.core.csv_text(rows))))


def main():
    with patch.object(t, "Predictor", side_effect=AssertionError("No inference in review")), \
         patch.object(t.core, "make_estimator", side_effect=AssertionError("No fitting in review")):
        data, features, refs, plan = c.validate()
        records, summaries = [], []
        protected = t.read_json(c.OUT/"preparation_audit.json")["protected_files"].copy()
        for seed in c.SEEDS:
            folder, ref = c.OUT/f"seed{seed}", refs[seed]
            s = c.read_completed(c.OUT, data, ref, plan, seed)
            summaries.append(s)
            binding = s["binding"]
            if binding["runtime"] != {"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 5060",
                                      "precision": "float32", "threads": 4, "telemetry": False, "hf_offline": True}:
                raise ValueError("Unexpected recorded runtime")
            pred = {n: np.full(len(data.y), np.nan) for n in (*t.REPS, t.PRIMARY)}
            choices, cell_seconds = [], []
            for fold in range(5):
                ti, vi = np.flatnonzero(ref["folds"] != fold), np.flatnonzero(ref["folds"] == fold)
                assert (len(ti), len(vi)) == (400, 100)
                splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold).split(ti, t.core.rank_groups(data.y[ti], 10)))
                inner = {n: np.full(400, np.nan) for n in t.REPS}
                split_records = []
                for rep in t.REPS:
                    for j, (tr, va) in enumerate(splits):
                        path = folder/f"fold{fold}_{rep}_inner{j}"
                        p = audit.cell(path, binding, rep, features[rep], data.y, ti[tr], ti[va], seed+1009*fold+j)
                        inner[rep][va] = p
                        split_records.append({"representation": rep, "inner_fold": j, "train_idx": ti[tr].tolist(),
                                              "valid_idx": ti[va].tolist(), "valid_mse": float(np.mean((p-data.y[ti[va]])**2))})
                        cell_seconds.append(t.read_json(path/"completed.json")["result"]["metadata"]["seconds"])
                    path = folder/f"fold{fold}_{rep}_outer"
                    pred[rep][vi] = audit.cell(path, binding, rep, features[rep], data.y, ti, vi, seed+1009*fold+9)
                    cell_seconds.append(t.read_json(path/"completed.json")["result"]["metadata"]["seconds"])
                choice, scores = t.select_representation(inner, data.y[ti])
                pred[t.PRIMARY][vi] = pred[choice][vi]
                choices.append({"fold": fold, "representation": choice, "inner_mse": scores, "inner_splits": split_records})
            assert choices == s["selection"]
            rebuilt, oof, diag, errors = c.build_results(data, ref, pred, choices, binding, seed, plan["frozen"]["common_failure_ids"])
            assert rebuilt == s
            for name, values in (("oof.csv", oof), ("diagnostics.csv", diag), ("per_image_errors.csv", errors)):
                assert audit.rows(folder/name) == csv_strings(values), name
            assert (folder/"report.md").read_text(encoding="utf-8") == c.report_text(s)
            current, blend = ref["current"], .5*(ref["current"]+pred[t.PRIMARY])
            gain = (current-data.y)**2-(blend-data.y)**2
            top = np.argsort(-gain, kind="stable")[:5]
            worst = set(np.argsort((blend-data.y)**2, kind="stable")[-20:].tolist())
            fixed = set(np.flatnonzero(np.isin(data.ids, plan["frozen"]["common_failure_ids"])).tolist())
            records.append({"seed": seed, "metrics": s["metrics"], "groups": s["groups"],
                            "choices": [r["representation"] for r in choices], "folds_improved": s["folds_improved"],
                            "group_gains": {g: s["groups"]["current"][g]-s["groups"][c.PRIMARY][g] for g in s["groups"]["current"]},
                            "common17_remaining_in_blend_worst20": len(fixed & worst),
                            "residual_correlation": float(np.corrcoef(current-data.y, pred[t.PRIMARY]-data.y)[0, 1]),
                            "improved_images": int((gain > 0).sum()), "net_sse_gain": float(gain.sum()),
                            "largest5_sse_gain": float(gain[top].sum()), "largest5_gain_ids": [str(data.ids[i]) for i in top],
                            "saved_cell_seconds_sum": float(sum(cell_seconds)), "cells": len(cell_seconds)})
        result = c.aggregate(summaries)
        result["protocol_sha256"] = t.common.sha256_file(c.OUT/"protocol.json")
        assert result == t.read_json(c.OUT/"confirmation_summary.json")
        assert audit.rows(c.OUT/"comparison.csv") == csv_strings(result["per_seed"])
        assert all(t.common.sha256_file(path) == sha for path, sha in protected.items())
        previous_count = len(protected)
        for path in c.OUT.rglob("*"):
            if path.is_file() and path.name != "review_results.json":
                protected[str(path.resolve())] = t.common.sha256_file(path)
        review = {"date_kst": "2026-10-10", "scope": "saved120 cell results only; no fitting/GPU inference/test images",
                  "checks": "frozen protocols, dependencies, features, same-fold baseline,120 context/query bindings, context target scaling, pooled inner selection, OOF, diagnostics, individual errors,50:50, aggregate, protected hashes",
                  "confirmation": result, "per_seed": records,
                  "mean_group_gain": {g: float(np.mean([r["group_gains"][g] for r in records]))
                                      for g in ("tail20", "middle80", "noiseQ4", "historical_common17", "current_worst20")},
                  "decision": "small fixed-blend gain direction supported; priority0.10 not met; close this tuning/split search; preserve as submission candidate while retaining original submission",
                  "next_candidate_step": "prepare full500 context with representation selected by fixed inner3-fold rule, test inference and separate50:50 candidate submission; not executed in this review",
                  "independent_holdout": False, "previous_protected_file_count": previous_count, "protected_files": protected,
                  "review_sources": [t.file_ref(__file__), t.file_ref(audit.__file__)]}
        t.write_json(c.OUT/"review_results.json", review)
    print("Audit passed:120 cells, all inner/outer partitions, choices, OOF/diagnostics,50:50, aggregate and protected files.")
    print("Mean group RMSE gains:", review["mean_group_gain"])
    for r in records:
        print(r["seed"], {k: r[k] for k in ("common17_remaining_in_blend_worst20", "residual_correlation", "improved_images", "net_sse_gain", "largest5_sse_gain", "saved_cell_seconds_sum")})
    print("Saved", c.OUT/"review_results.json")


if __name__ == "__main__":
    main()
