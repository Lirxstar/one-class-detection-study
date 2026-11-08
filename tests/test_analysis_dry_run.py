"""Analysis, figures, and note builder run end to end on fabricated random predictions
(temporary directory, final-run schema). Exercises the code before any final run; no
real score is read."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

GENS = ["SD15", "ADM", "GLIDE", "Midjourney"]
METHODS = ["B1", "O1", "O2", "M0", "B1-C1", "O1-k1", "O1-k10", "M0-L2", "M0-ratio"]


def _block(rng, exp, backbone, cond, pert, enc, methods, n_real=40, n_fake=20, cal=True):
    rows = []
    ids_real = [f"validation/v/{i}" for i in range(n_real)]
    ids_cal = [f"validation/c/{i}" for i in range(n_real)]
    ids_fake = {g: [f"validation/{g}/{i}" for i in range(n_fake)] for g in GENS}
    for m in methods:
        tau = 0.5
        parts = ([("R_cal", 0, "Real", ids_cal)] if cal else []) + [("R_test", 0, "Real", ids_real)] + \
                [("F_seen_test" if g == "SD15" else "F_unseen_test", 1, g, ids_fake[g]) for g in GENS]
        for part, lab, gen, ids in parts:
            s = rng.normal(lab * 0.8, 1, len(ids))
            rows.append(pd.DataFrame({"experiment": exp, "backbone": backbone, "data_condition": cond,
                                      "perturbation": pert, "encoding_condition": enc, "method": m, "row_id": ids,
                                      "partition": part, "authenticity_label": lab, "generator": gen, "score": s,
                                      "threshold": tau, "predicted_label": (s > tau).astype(int), "selected_C": 1.0}))
    return pd.concat(rows)


@pytest.fixture()
def fake_preds(tmp_path, monkeypatch):
    import calibrate
    rng = np.random.default_rng(0)
    pred = tmp_path / "predictions"
    for d in ("e1", "e2", "e3"):
        (pred / d).mkdir(parents=True)
    e1 = pd.concat([_block(rng, "E1", b, c, "clean", "none", METHODS) for b, c in
                    [("primary", "d_ctrl"), ("primary", "d_raw"), ("secondary", "d_ctrl")]])
    # thresholds consistent with the calibration rule
    for key, g in e1[e1["partition"] == "R_cal"].groupby(["backbone", "data_condition", "method"]):
        tau = calibrate.threshold(g["score"])
        m = (e1["backbone"] == key[0]) & (e1["data_condition"] == key[1]) & (e1["method"] == key[2])
        e1.loc[m, "threshold"] = tau
        e1.loc[m, "predicted_label"] = (e1.loc[m, "score"] > tau).astype(int)
    e1.to_csv(pred / "e1" / "e1_all.csv.gz", index=False)
    json.dump({"primary_d_ctrl": {"B1_selected_C": 1.0, "O2_shrinkage": 0.1, "O2_condition_number": 10.0}},
              open(pred / "e1" / "e1_diagnostics.json", "w"))
    e2 = pd.concat([_block(rng, "E2", "primary", c, t, "none", METHODS, cal=False) for c in ("d_ctrl", "d_raw")
                    for t in ["jpeg95", "jpeg75", "jpeg50", "blur0.5", "blur1.0"]])
    e2.to_csv(pred / "e2" / "e2_all.csv.gz", index=False)
    e3b = _block(rng, "E3B", "primary", "d_jpeg", "clean", "none", METHODS)
    for key, g in e3b[e3b["partition"] == "R_cal"].groupby("method"):
        tau = calibrate.threshold(g["score"])
        m = e3b["method"] == key
        e3b.loc[m, "threshold"] = tau
        e3b.loc[m, "predicted_label"] = (e3b.loc[m, "score"] > tau).astype(int)
    e3b.to_csv(pred / "e3" / "e3b_primary_d_jpeg.csv.gz", index=False)
    meta = _block(rng, "E3A", "none", "stored_file", "clean", "none", ["META-primary", "META-supplementary"])
    meta.to_csv(pred / "e3" / "e3a_metadata.csv.gz", index=False)
    e3c = pd.concat([_block(rng, "E3C", "primary", "d_ctrl", "clean", v, ["B1", "O1", "O2", "M0"], cal=False)
                     for v in ["C0_common", "C1_biased", "C1_matched", "C1_inverted"]])
    e3c.to_csv(pred / "e3" / "e3c_primary_d_ctrl.csv.gz", index=False)
    return tmp_path


def test_analysis_and_figures_dry_run(fake_preds, monkeypatch):
    import analysis
    import figures
    importlib.reload(analysis)
    monkeypatch.setattr(analysis, "PRED", fake_preds / "predictions")
    monkeypatch.setattr(analysis, "TAB", fake_preds / "tables")
    monkeypatch.setattr(analysis, "_DRAWS", None)
    analysis.main()
    vals = json.loads((fake_preds / "tables" / "note_values.json").read_text())
    assert "table1_d_ctrl.O1.auroc_ADM" in vals and "ladder.d_ctrl.M0.rank" in vals and "e3c.B1.C1_biased.fpr" in vals
    for name in ("table1_d_ctrl", "table2_e2_fpr_d_ctrl", "table3_ladder", "e3a_metadata", "e3c_injected_cue"):
        assert (fake_preds / "tables" / f"{name}.md").exists()
    importlib.reload(figures)
    monkeypatch.setattr(figures, "A", analysis)
    monkeypatch.setattr(figures, "OUT", fake_preds / "figures")
    figures.main()
    assert len(list((fake_preds / "figures").glob("*.svg"))) == 9
    per_gen = pd.read_csv(fake_preds / "tables" / "e2_per_generator_d_ctrl.csv")
    assert set(per_gen["generator"]) == set(GENS)
    assert set(per_gen["transform"]) == {"clean", "jpeg95", "jpeg75", "jpeg50", "blur0.5", "blur1.0"}
    assert {"auroc_ci_lo", "auroc_ci_hi", "tpr_ci_lo", "tpr_ci_hi", "tp", "fp"} <= set(per_gen)
    contrasts = pd.read_csv(fake_preds / "tables" / "e3c_within_c1.csv")
    assert len(contrasts) == 4 * 3 * 11
    assert not contrasts["variant_a"].str.startswith("C0").any()
    # The published contrast must equal the difference of its two aligned point estimates.
    a = vals["e3c.B1.C1_biased.mean_unseen_auroc"]
    b = vals["e3c.B1.C1_matched.mean_unseen_auroc"]
    assert vals["e3c_pair.B1.C1_biased-C1_matched.mean_unseen_auroc"] == pytest.approx(a - b)
