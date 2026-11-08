#!/usr/bin/env python3
"""E3 final (W8): E3A metadata baseline, E3B D-jpeg clean evaluation (ladder with E1),
E3C injected JPEG cue on D-ctrl views. Refuses to overwrite existing results in the public release."""
import json

import metadata_baseline
import pipeline as P
from config import ROOT

OUT = P.PRED_DIR / "e3"


def main():
    outs = [OUT / "e3a_metadata.csv.gz", OUT / "e3b_primary_d_jpeg.csv.gz", OUT / "e3c_primary_d_ctrl.csv.gz",
            OUT / "e3_diagnostics.json"]
    commit = P.assert_new_run_allowed(outs)
    run_id = P.new_run_id("e3")
    diag = {"run_id": run_id, "git_commit": commit}
    meta, diag["e3a"] = metadata_baseline.run("final", "E3A", run_id, commit, include_supplementary=True)
    P.save(meta, outs[0])
    preds, _, _, diag["e3b_d_jpeg"] = P.run_clean("final", "primary", "d_jpeg", "E3B", run_id)
    P.save(preds, outs[1])
    c0 = json.loads((ROOT / "artifacts" / "audit" / "c0_png_equality.json").read_text())["all_bitwise_equal"]
    e3c, diag["e3c"] = P.run_e3c("final", "E3C", run_id, c0)
    P.save(e3c, outs[2])
    outs[-1].write_text(json.dumps(diag, indent=1, default=float) + "\n")
    print("saved E3A, E3B, E3C predictions")


if __name__ == "__main__":
    main()
