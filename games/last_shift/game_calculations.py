"""Deterministic Last Shift mechanics used by the SDK game state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from games.last_shift.game_config import GameConfig


Board = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class WinningGroup:
    """Authoritative scatter-pay group and all payout-contributing positions."""

    symbol: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class CargoSelection:
    column: int
    source_positions: tuple[int, ...]
    regular_winning_count: int
    selection_rank: int


@dataclass(frozen=True)
class LoadTransition:
    column: int
    level_before: int
    level_after: int
    source_positions: tuple[int, ...]
    regular_winning_count: int
    selection_rank: int


@dataclass(frozen=True)
class SymbolChange:
    column: int
    position: int
    from_symbol: str
    to_symbol: str


@dataclass(frozen=True)
class DepartureGroup:
    departure_id: str
    kind: str
    columns: tuple[int, ...]
    components: tuple[int, ...]
    multiplier: int
    stage_at_selection: str


def validate_board(board: Board, config: GameConfig) -> None:
    if len(board) != config.num_reels:
        raise ValueError("board must have exactly six columns")
    if any(len(column) != config.num_rows[index] for index, column in enumerate(board)):
        raise ValueError("each board column must have exactly five rows")
    valid = set(config.regular_symbols) | {"W", "S"}
    if any(symbol not in valid for column in board for symbol in column):
        raise ValueError("board contains an unknown symbol")


def flat_position(column: int, row: int, columns: int = 6) -> int:
    return row * columns + column


def coordinates(position: int, columns: int = 6, rows: int = 5) -> tuple[int, int]:
    if position < 0 or position >= columns * rows:
        raise ValueError(f"position {position} is outside the board")
    return position % columns, position // columns


def symbol_at(board: Board, position: int) -> str:
    column, row = coordinates(position, len(board), len(board[0]))
    return board[column][row]


def select_cargo_columns(
    board: Board,
    winning_groups: Sequence[WinningGroup],
    config: GameConfig,
) -> tuple[CargoSelection, ...]:
    """Select at most two columns from regular winning symbols only."""
    validate_board(board, config)
    positions_by_column: dict[int, set[int]] = defaultdict(set)
    for group in winning_groups:
        if group.symbol not in config.regular_symbols:
            raise ValueError(f"{group.symbol} is not a regular paying symbol")
        for position in group.positions:
            column, _ = coordinates(position, config.num_reels, config.num_rows[0])
            if symbol_at(board, position) == group.symbol:
                positions_by_column[column].add(position)

    qualified = [
        (column, tuple(sorted(positions)))
        for column, positions in positions_by_column.items()
        if len(positions) >= config.minimum_cargo_count
    ]
    qualified.sort(key=lambda item: (-len(item[1]), item[0]))
    return tuple(
        CargoSelection(
            column=column,
            source_positions=positions,
            regular_winning_count=len(positions),
            selection_rank=rank,
        )
        for rank, (column, positions) in enumerate(
            qualified[: config.maximum_columns_loaded], start=1
        )
    )


def load_selected_columns(
    levels: Sequence[int], selections: Sequence[CargoSelection]
) -> tuple[tuple[int, ...], tuple[LoadTransition, ...]]:
    if len(levels) != 6 or any(level not in (0, 1, 2) for level in levels):
        raise ValueError("stable column levels must be six values in range 0..2")
    if len(selections) > 2:
        raise ValueError("an evaluation cannot load more than two columns")
    selected_columns = [selection.column for selection in selections]
    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError("selected columns must be unique")

    next_levels = list(levels)
    transitions = []
    for selection in selections:
        before = next_levels[selection.column]
        after = min(before + 1, 3)
        if after < 3:
            next_levels[selection.column] = after
        transitions.append(
            LoadTransition(
                column=selection.column,
                level_before=before,
                level_after=after,
                source_positions=selection.source_positions,
                regular_winning_count=selection.regular_winning_count,
                selection_rank=selection.selection_rank,
            )
        )
    return tuple(next_levels), tuple(transitions)


def apply_level_modifier(
    board: Board, transition: LoadTransition, config: GameConfig
) -> tuple[Board, tuple[SymbolChange, ...]]:
    """Apply the 1-wild/2-wild ladder for the next evaluation."""
    if transition.level_after not in (1, 2):
        raise ValueError("level modifier is defined only for levels one and two")
    validate_board(board, config)
    eligible = [
        flat_position(transition.column, row, config.num_reels)
        for row, symbol in enumerate(board[transition.column])
        if symbol in config.regular_symbols
    ]
    required = transition.level_after
    if len(eligible) < required:
        raise ValueError("deterministic board cannot satisfy the scheduled modifier")

    mutable = [list(column) for column in board]
    changes = []
    for position in eligible[:required]:
        column, row = coordinates(position, config.num_reels, config.num_rows[0])
        previous = mutable[column][row]
        mutable[column][row] = "W"
        changes.append(SymbolChange(column, position, previous, "W"))
    return tuple(tuple(column) for column in mutable), tuple(changes)


def group_completed_columns(columns: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted(set(columns)))
    if len(ordered) > 2:
        raise ValueError("at most two columns can complete in one evaluation")
    if any(column < 0 or column >= 6 for column in ordered):
        raise ValueError("completed column is outside the board")
    if len(ordered) == 2 and ordered[1] == ordered[0] + 1:
        return (ordered,)
    return tuple((column,) for column in ordered)


def make_departure_groups(
    completed_columns: Iterable[int],
    stage: str,
    component_by_column: dict[int, int],
    departure_sequence: int,
) -> tuple[DepartureGroup, ...]:
    if stage not in ("yard", "mainline", "redline"):
        raise ValueError("unknown departure stage")
    groups = []
    for offset, columns in enumerate(group_completed_columns(completed_columns)):
        components = tuple(component_by_column[column] for column in columns)
        groups.append(
            DepartureGroup(
                departure_id=f"dep-{departure_sequence + offset:04d}",
                kind="coupled" if len(columns) == 2 else "single",
                columns=columns,
                components=components,
                multiplier=sum(components),
                stage_at_selection=stage,
            )
        )
    return tuple(groups)


def apply_departure_modifier(
    board: Board, groups: Sequence[DepartureGroup], config: GameConfig
) -> tuple[Board, tuple[SymbolChange, ...]]:
    validate_board(board, config)
    mutable = [list(column) for column in board]
    changes = []
    for group in groups:
        for column in group.columns:
            for row, previous in enumerate(mutable[column]):
                position = flat_position(column, row, config.num_reels)
                mutable[column][row] = "W"
                changes.append(SymbolChange(column, position, previous, "W"))
    return tuple(tuple(column) for column in mutable), tuple(changes)
