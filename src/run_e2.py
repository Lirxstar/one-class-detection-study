#!/usr/bin/env python3
"""E2 final (W7): JPEG and blur on R_test and synthetic test views, D-ctrl and D-raw,
with the E1 models and clean thresholds. Refuses to overwrite existing results in the public release.
Models are refit deterministically from the same features; selected C and every threshold
must equal the values stored in the E1 predictions, otherwise the run aborts."""
import json

import numpy as np
import pandas as pd

import conditions as C
import pipeline as P

OUT = P.PRED_DIR / "e2"


def main():
    outs = [OUT / f"e2_primary_{c}.csv.gz" for c in ("d_ctrl", "d_raw")] + [OUT / "e2_diagnostics.json"]
    commit = P.assert_new_run_allowed(outs)
    run_id = P.new_run_id("e2")
    diag = {"run_id": run_id, "git_commit": commit, "verified_against_e1": {}}
    for cond, path in zip(("d_ctrl", "d_raw"), outs):
        e1 = pd.read_csv(P.PRED_DIR / "e1" / f"e1_primary_{cond}.csv.gz")
        _, fit, taus, _ = P.run_clean("final", "primary", cond, "E2-refit-check", run_id)
        stored = e1.groupby("method")["threshold"].first().to_dict()
        assert set(stored) == set(taus) and all(np.float64(stored[m]) == np.float64(taus[m]) for m in taus), "thresholds differ from E1"
        assert float(e1.loc[e1["method"] == "B1", "selected_C"].iloc[0]) == fit.b1.C, "B1 C differs from E1"
        diag["verified_against_e1"][cond] = True
        preds = P.run_e2("final", cond, fit, taus, "E2", run_id, C.E2_NAMES[1:])
        P.save(preds, path)
        print(f"saved {path.name}: {len(preds)} rows", flush=True)
    outs[-1].write_text(json.dumps(diag, indent=1) + "\n")


if __name__ == "__main__":
    main()
