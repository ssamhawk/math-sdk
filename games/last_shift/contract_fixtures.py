"""Immutable boards used only to prove the T4-C1G SDK contract."""

LOSS_BOARD = tuple(
    tuple("ABCDEFGH"[(row * 6 + column) % 8] for row in range(5))
    for column in range(6)
)

CARGO_BOARD = (
    tuple("BCDEF"),
    tuple("GHBCD"),
    tuple("AAAAA"),
    tuple("AAAAA"),
    tuple("EFGHB"),
    tuple("CDEFG"),
)
