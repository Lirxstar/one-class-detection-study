#!/usr/bin/env python3
"""E1 final (W6): clean evaluation in D-ctrl and D-raw (ViT-S/14) and D-ctrl (ViT-B/14, secondary).
Refuses to overwrite existing results in the public release."""
import json

import pipeline as P

OUT = P.PRED_DIR / "e1"
JOBS = [("primary", "d_ctrl"), ("primary", "d_raw"), ("secondary", "d_ctrl")]


def main():
    outs = [OUT / f"e1_{b}_{c}.csv.gz" for b, c in JOBS] + [OUT / "e1_diagnostics.json"]
    commit = P.assert_new_run_allowed(outs)
    run_id = P.new_run_id("e1")
    diag = {"run_id": run_id, "git_commit": commit}
    for (b, c), path in zip(JOBS, outs):
        preds, _, _, d = P.run_clean("final", b, c, "E1", run_id)
        P.save(preds, path)
        diag[f"{b}_{c}"] = d
        print(f"saved {path.name}: {len(preds)} rows", flush=True)
    outs[-1].write_text(json.dumps(diag, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
