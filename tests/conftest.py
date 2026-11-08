import json

import pandas as pd
import pytest

from config import ROOT

TRAIN_PARTS = {"R_train", "F_seen_train", "P_cal", "P_test", "P_fake"}
VAL_PARTS = {"R_cal", "R_test", "F_seen_test", "F_unseen_test"}


def flat(v):
    return [r for lst in v.values() for r in lst] if isinstance(v, dict) else list(v)


@pytest.fixture(scope="session")
def splits():
    return json.loads((ROOT / "artifacts" / "splits.json").read_text())


@pytest.fixture(scope="session")
def manifest():
    return pd.read_csv(ROOT / "artifacts" / "manifest.csv", keep_default_na=False, na_values=[""],
                       dtype={"partition": str, "exclusion_reason": str})


@pytest.fixture(scope="session")
def parts(splits):
    return {k: flat(v) for k, v in splits["partitions"].items()}


@pytest.fixture(scope="session")
def view_index():
    p = ROOT / "artifacts" / "cache" / "views" / "index.csv"
    if not p.exists():
        pytest.skip("view cache not built (run src/make_conditions.py)")
    return pd.read_csv(p)
