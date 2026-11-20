#!/usr/bin/env python3
"""Classify every prompt in the test set against a policy version.

    export ANTHROPIC_API_KEY=...
    python eval/run_eval.py --policy policy/policy_v1.yaml
    python eval/run_eval.py --policy policy/policy_v1.yaml --mock   # dry run, no API

For each prompt, Claude returns JSON with clause_id, decision
(allow/label/limit/escalate), confidence (0-1) and a one-sentence rationale
citing the clause. Results go to eval/results_<policy_version>.csv.

A refusal or an API error is recorded in the stop_reason / error columns and
is never silently dropped. report.py counts these rows separately.
"""

from __future__ import annotations

import argparse
import sys

from safenet_common import (RESULT_FIELDS, ROOT, add_llm_args, llm_from_args,
                            load_policy, make_classifier, read_csv, run_concurrent,
                            write_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="path to policy YAML")
    ap.add_argument("--test-set", default=str(ROOT / "data" / "test_set.csv"))
    ap.add_argument("--out", help="output CSV (default eval/results_<policy_version>.csv)")
    ap.add_argument("--limit", type=int, help="only classify the first N prompts")
    add_llm_args(ap)
    args = ap.parse_args()

    policy = load_policy(args.policy)
    rows = [r for r in read_csv(args.test_set) if r.get("prompt")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit(f"No prompts in {args.test_set}. Add rows (see scripts/generate_candidates.py --promote).")

    classify = make_classifier(policy, llm_from_args(args))
    print(f"Classifying {len(rows)} prompts against policy {policy.version} "
          f"with {'mock' if args.mock else args.model}…", file=sys.stderr)
    preds = run_concurrent(rows, lambda r: classify(r["prompt"]), args.workers, "prompts")

    out_rows = [{**r, **p} for r, p in zip(rows, preds)]
    out = args.out or str(ROOT / "eval" / f"results_{policy.version}.csv")
    write_csv(out, out_rows, RESULT_FIELDS)

    n_ref = sum(p["stop_reason"] == "refusal" for p in preds)
    n_err = sum(bool(p["error"]) and p["stop_reason"] != "refusal" for p in preds)
    n_nocite = sum(p["cites_clause"] == "n" for p in preds)
    print(f"Wrote {out}  ({len(preds)} rows, {n_ref} refusals, {n_err} errors, "
          f"{n_nocite} rationales not citing their clause_id)", file=sys.stderr)


if __name__ == "__main__":
    main()
