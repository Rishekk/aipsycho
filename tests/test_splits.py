import pytest

from src.data_processing.splits import (
    RAVDESS_TEST_ACTORS, RAVDESS_TRAIN_ACTORS, RAVDESS_VAL_ACTORS,
    actors_in, holdout_split, ravdess_actor_id, split_ravdess_files,
)


def _all_ravdess_files():
    """Every actor x 2 emotions x 2 repetitions, using the real filename convention."""
    files = []
    for actor in range(1, 25):
        for emotion in ("01", "05"):
            for rep in ("01", "02"):
                files.append(f"Actor_{actor:02d}/03-01-{emotion}-01-01-{rep}-{actor:02d}.wav")
    return files


def test_actor_id_parsing():
    assert ravdess_actor_id("03-01-06-01-02-01-12.wav") == 12
    assert ravdess_actor_id("/x/Actor_07/03-01-01-01-01-01-07.wav") == 7
    assert ravdess_actor_id("not_ravdess.wav") is None
    assert ravdess_actor_id("03-01-06-01-02-01-xx.wav") is None


def test_actor_groups_are_disjoint_and_cover_all_24():
    tr, va, te = set(RAVDESS_TRAIN_ACTORS), set(RAVDESS_VAL_ACTORS), set(RAVDESS_TEST_ACTORS)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(1, 25))


def test_heldout_actors_are_gender_balanced():
    # RAVDESS: odd actor id = male, even = female.
    for group in (RAVDESS_VAL_ACTORS, RAVDESS_TEST_ACTORS):
        assert sum(a % 2 for a in group) == len(group) // 2


def test_no_speaker_appears_in_two_splits():
    files = _all_ravdess_files()
    train = split_ravdess_files(files, "train")
    val = split_ravdess_files(files, "val")
    test = split_ravdess_files(files, "test")
    a_tr, a_va, a_te = actors_in(train), actors_in(val), actors_in(test)
    assert not (a_tr & a_va) and not (a_tr & a_te) and not (a_va & a_te)
    # Nothing lost, nothing duplicated.
    assert sorted(train + val + test) == sorted(files)


def test_split_is_deterministic_and_independent_of_input_order():
    files = _all_ravdess_files()
    assert split_ravdess_files(files, "test") == split_ravdess_files(list(reversed(files)), "test")


def test_unparseable_filenames_are_dropped_not_assigned():
    files = _all_ravdess_files() + ["mystery.wav", "03-01-01-01-01-01-zz.wav"]
    for split in ("train", "val", "test"):
        assert not any("mystery" in p or "zz" in p for p in split_ravdess_files(files, split))


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        split_ravdess_files(["03-01-01-01-01-01-01.wav"], "dev")


def test_holdout_split_is_disjoint_complete_and_deterministic():
    items = [f"img_{i:03d}.png" for i in range(200)]
    train = holdout_split(items, "train")
    val = holdout_split(items, "val")
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(items)
    assert len(val) == 20
    assert holdout_split(items, "val") == val
    assert holdout_split(list(reversed(items)), "val") == val


def test_holdout_split_rejects_test_split():
    with pytest.raises(ValueError):
        holdout_split(["a", "b"], "test")
