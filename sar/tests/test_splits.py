from splits import (
    BlockGrid,
    assign_blocks,
    assert_no_train_test_overlap,
    bboxes_intersect,
)


def test_straddle_and_centre_assignment():
    grid = BlockGrid(origin_x=0.0, origin_y=0.0, size=1000.0)
    # 128 m tile fully inside block (0,0)
    assert grid.block_id(500, 500) == (0, 0)
    assert not grid.straddles(436, 436, 564, 564)
    # tile sitting on the 1000 m line
    assert grid.straddles(940, 400, 1068, 528)


def test_west_east_split_ratios():
    blocks = [(x, 0) for x in range(10)]
    mapping = assign_blocks(blocks, (0.70, 0.15, 0.15))
    counts = {"train": 0, "val": 0, "test": 0}
    for b in blocks:
        counts[mapping[b]] += 1
    assert counts["train"] == 7
    assert counts["val"] == 2
    assert counts["test"] == 1
    # west is train, east is test
    assert mapping[(0, 0)] == "train"
    assert mapping[(9, 0)] == "test"


def test_train_test_bboxes_do_not_intersect():
    train = [(0, 0, 100, 100), (200, 0, 300, 100)]
    test = [(400, 0, 500, 100)]
    assert_no_train_test_overlap(train, test)
    try:
        assert_no_train_test_overlap(train, [(90, 0, 150, 50)])
    except AssertionError:
        return
    raise AssertionError("expected overlap to fail")


def test_bboxes_intersect_touching_edge_is_not_overlap():
    # half-open style: sharing an edge is not an interior intersection
    # our test uses strict < so a shared edge at 100 does not count
    assert not bboxes_intersect((0, 0, 100, 100), (100, 0, 200, 100))
