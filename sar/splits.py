"""Geographic block assignment. Not random, not by strip.

SpaceNet 6 strips overlap: the same building is in many collects. A tile
belongs to the 1 km block that contains its centre. Tiles that straddle a
block edge are discarded so a train tile cannot leak across the cut.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockGrid:
    origin_x: float
    origin_y: float
    size: float = 1000.0

    def block_id(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.origin_x) // self.size)
        iy = int((y - self.origin_y) // self.size)
        return (ix, iy)

    def block_bounds(self, block: tuple[int, int]) -> tuple[float, float, float, float]:
        ix, iy = block
        west = self.origin_x + ix * self.size
        south = self.origin_y + iy * self.size
        return (west, south, west + self.size, south + self.size)

    def straddles(self, west: float, south: float, east: float, north: float) -> bool:
        corners = (
            (west, south),
            (east, south),
            (west, north),
            (east, north),
        )
        ids = {self.block_id(x, y) for x, y in corners}
        return len(ids) > 1


def assign_blocks(
    blocks: list[tuple[int, int]],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[tuple[int, int], str]:
    """West→east: train, then val, then test, by block count ≈ area."""
    if not blocks:
        return {}
    ordered = sorted(blocks)  # ix, iy — west to east, then south to north
    n = len(ordered)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(0, int(round(n * ratios[1])))
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    n_train = min(n_train, n - (1 if n > 1 else 0))
    mapping: dict[tuple[int, int], str] = {}
    for i, block in enumerate(ordered):
        if i < n_train:
            mapping[block] = "train"
        elif i < n_train + n_val:
            mapping[block] = "val"
        else:
            mapping[block] = "test"
    return mapping


def bboxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def assert_no_train_test_overlap(
    train: list[tuple[float, float, float, float]],
    test: list[tuple[float, float, float, float]],
) -> None:
    for tb in train:
        for ub in test:
            if bboxes_intersect(tb, ub):
                raise AssertionError(f"train bbox {tb} intersects test bbox {ub}")
