"""Labels and generator roles in the manifest and partitions (PROTOCOL.md 2.2)."""
import config

CORE = config.load("core")["data"]
EXPECTED = {
    "R_train": {"Real"}, "R_cal": {"Real"}, "R_test": {"Real"}, "P_cal": {"Real"}, "P_test": {"Real"},
    "F_seen_train": {CORE["generator_seen"]}, "F_seen_test": {CORE["generator_seen"]},
    "F_unseen_test": set(CORE["generators_unseen"]), "P_fake": set(CORE["generators_dev_only"]) - {"BigGAN"},
}


def test_label_matches_generator(manifest):
    assert ((manifest["label"] == 0) == (manifest["generator"] == "Real")).all()


def test_sd14_absent_and_seen_is_sd15(manifest, splits):
    assert "SD14" not in set(manifest["generator"])
    assert CORE["generator_seen"] == "SD15" and splits["generator_seen"] == "SD15"


def test_partition_generators(manifest, parts):
    gen = manifest.set_index("row_id")["generator"]
    for p, ids in parts.items():
        assert set(gen.loc[ids]) == EXPECTED[p], p


def test_unseen_and_dev_generators_never_in_training_partitions(manifest, parts):
    gen = manifest.set_index("row_id")["generator"]
    for p in ("R_train", "F_seen_train"):
        assert set(gen.loc[parts[p]]).isdisjoint(set(CORE["generators_unseen"]) | set(CORE["generators_dev_only"]))


def test_partition_column_agrees_with_splits(manifest, parts):
    m = manifest[manifest["partition"].notna() & (manifest["partition"] != "")]
    assert len(m) == sum(len(v) for v in parts.values())
    by = m.groupby("partition")["row_id"].apply(set).to_dict()
    assert by == {k: set(v) for k, v in parts.items()}


def test_counts(splits):
    c = splits["counts"]
    assert c["R_train"] == 2000 and c["R_cal"] == 500 and c["R_test"] == 500
    assert c["F_seen_train"] == splits["eligible_pool_sizes"]["train_seen"] <= 2000
    assert c["F_seen_test"] == 300 and c["F_unseen_test"] == {"ADM": 300, "GLIDE": 300, "Midjourney": 300}
