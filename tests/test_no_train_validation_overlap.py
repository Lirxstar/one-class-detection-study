"""Disjointness and leakage rules (PROTOCOL.md 2.4, 2.5)."""
import itertools

from conftest import TRAIN_PARTS, VAL_PARTS


def test_partitions_pairwise_disjoint(parts):
    for a, b in itertools.combinations(parts, 2):
        assert not set(parts[a]) & set(parts[b]), (a, b)
    for p, ids in parts.items():
        assert len(ids) == len(set(ids)), p


def test_partitions_respect_split_boundary(manifest, parts):
    split = manifest.set_index("row_id")["split"]
    for p in TRAIN_PARTS:
        assert set(split.loc[parts[p]]) <= {"train"}, p
    for p in VAL_PARTS:
        assert set(split.loc[parts[p]]) == {"validation"}, p


def test_no_duplicate_group_in_two_partitions_or_twice(manifest, parts):
    m = manifest.set_index("row_id")
    used = [r for ids in parts.values() for r in ids]
    groups = m.loc[used, "dup_group"]
    assert groups.is_unique, "a duplicate group contributes more than one partition row"
    assert not m.loc[used, "group_crosses_split"].any()
    assert not m.loc[used, "group_crosses_label"].any()


def test_validation_groups_have_no_train_member(manifest, parts):
    val_groups = set(manifest.set_index("row_id").loc[[r for p in VAL_PARTS for r in parts[p]], "dup_group"])
    train_groups = set(manifest.loc[manifest["split"] == "train", "dup_group"])
    assert not val_groups & train_groups


def test_cal_and_test_real_disjoint(parts):
    assert not set(parts["R_cal"]) & set(parts["R_test"])
    assert not set(parts["P_cal"]) & set(parts["R_train"])
