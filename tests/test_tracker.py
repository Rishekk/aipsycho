from app.multi_person_tracker import PersonTracker, _iou


def test_iou_identical_disjoint_and_partial():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 5, 10, 10)) < 1.0


def test_ids_are_stable_across_small_motion():
    t = PersonTracker(iou_threshold=0.25)
    first = t.update([(100, 100, 50, 50)])
    second = t.update([(104, 102, 50, 50)])
    assert first[0][0] == second[0][0] == 1


def test_two_people_get_two_ids_and_keep_them():
    t = PersonTracker()
    a = dict(t.update([(0, 0, 50, 50), (200, 0, 50, 50)]))
    b = dict(t.update([(2, 1, 50, 50), (203, 2, 50, 50)]))
    assert set(a) == set(b) == {1, 2}


def test_track_expires_after_max_lost_frames():
    t = PersonTracker(max_lost_frames=3)
    t.update([(0, 0, 50, 50)])
    for _ in range(5):
        t.update([])
    assert t.get_person(1) is None
    assert t.update([(0, 0, 50, 50)])[0][0] == 2  # new identity, not a silent reuse


def test_reset_restarts_ids():
    t = PersonTracker()
    t.update([(0, 0, 50, 50)])
    t.reset()
    assert t.update([(0, 0, 50, 50)])[0][0] == 1


def test_tracked_person_has_no_demo_state():
    t = PersonTracker()
    t.update([(0, 0, 50, 50)])
    assert not hasattr(t.get_person(1), "next_demo_state")
