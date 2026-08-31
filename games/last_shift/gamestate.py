"""Last Shift round and bonus state transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Sequence

from games.last_shift.game_calculations import (
    Board,
    DepartureGroup,
    derive_scatter_positions,
    validate_board,
)
from games.last_shift.game_config import GameConfig
from games.last_shift.game_events import EventLedger


STAGES = ("yard", "mainline", "redline")
RESET_LEVELS = {"yard": 0, "mainline": 1, "redline": 2}


def stage_for_departures(departures: int) -> str:
    if departures >= 4:
        return "redline"
    if departures >= 2:
        return "mainline"
    return "yard"


def validate_trigger_positions(
    board: Board, positions: Sequence[int], config: GameConfig
) -> tuple[int, ...]:
    validate_board(board, config)
    normalized = tuple(positions)
    if len(normalized) < 4:
        raise ValueError("bonus trigger requires at least four positions")
    if len(set(normalized)) != len(normalized):
        raise ValueError("bonus trigger positions must be unique")
    if any(not isinstance(position, int) or position < 0 or position >= 30 for position in normalized):
        raise ValueError("bonus trigger position is outside the board")
    authoritative = derive_scatter_positions(board, config)
    if tuple(sorted(normalized)) != authoritative:
        raise ValueError("bonus trigger positions must point to S and equal the complete sorted scatter set")
    return normalized


@dataclass(frozen=True)
class RoundState:
    mode: str = "basegame"
    spin_index: int = 0
    cascade_index: int = 0
    column_levels: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    departures: int = 0
    stage: str = "yard"
    free_spins_remaining: int = 0
    round_payout_units: int = 0
    capped: bool = False

    def __post_init__(self):
        if self.mode not in ("basegame", "freegame"):
            raise ValueError("mode must be basegame or freegame")
        if len(self.column_levels) != 6 or any(
            level not in (0, 1, 2) for level in self.column_levels
        ):
            raise ValueError("stable levels must contain six values in range 0..2")
        if self.stage not in STAGES:
            raise ValueError("unknown bonus stage")
        if min(
            self.spin_index,
            self.cascade_index,
            self.departures,
            self.free_spins_remaining,
            self.round_payout_units,
        ) < 0:
            raise ValueError("round state counters cannot be negative")
        if self.mode == "basegame" and (
            self.departures != 0 or self.stage != "yard" or self.free_spins_remaining != 0
        ):
            raise ValueError("bonus-only state cannot leak into basegame")


class LastShiftStateMachine:
    """Deterministic state facade that can be called by an SDK GameState."""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random()
        self.active_outcome_path = "natural"

    def reset_seed(self, sim: int, simulation_seed: int | None = None) -> int:
        """Match SDK simulation seeding while keeping RNG local to this game."""
        seed = (simulation_seed if simulation_seed is not None else sim) + 1
        self.rng.seed(seed)
        return seed

    def deterministic_trace(self, length: int = 8) -> tuple[int, ...]:
        """Small reproducibility trace; this is not a tuned reel draw."""
        if length < 1:
            raise ValueError("trace length must be positive")
        return tuple(self.rng.randrange(2**31) for _ in range(length))

    def new_base_round(self) -> tuple[RoundState, EventLedger]:
        self.active_outcome_path = "natural"
        state = RoundState()
        ledger = EventLedger(
            self.config.wincap_units,
            payout_quantum_units=self.config.payout_quantum_units,
        )
        ledger.round_start(
            mode=state.mode,
            levels=state.column_levels,
            departures=state.departures,
            stage=state.stage,
            free_spins_remaining=state.free_spins_remaining,
        )
        return state, ledger

    def trigger_natural_bonus(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
    ) -> RoundState:
        if state.mode != self.config.basegame_type:
            raise ValueError("natural bonus can only trigger from basegame")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        bonus_state = RoundState(
            mode=self.config.freegame_type,
            column_levels=state.column_levels,
            departures=0,
            stage="yard",
            free_spins_remaining=self.config.initial_free_spins,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )
        ledger.bonus_trigger(
            positions,
            self.config.initial_free_spins,
            bonus_state.column_levels,
            outcome_path="natural",
            board=board,
        )
        self.active_outcome_path = "natural"
        return bonus_state

    def build_forced_bonus(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
        criterion: str,
    ) -> RoundState:
        if criterion not in self.config.outcome_paths["forced"]:
            raise ValueError("forced outcome requires a dedicated forced criterion")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        bonus_state = RoundState(
            mode=self.config.freegame_type,
            column_levels=state.column_levels,
            departures=0,
            stage="yard",
            free_spins_remaining=self.config.initial_free_spins,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )
        ledger.bonus_trigger(
            positions,
            self.config.initial_free_spins,
            bonus_state.column_levels,
            outcome_path="forced",
            board=board,
        )
        self.active_outcome_path = "forced"
        return bonus_state

    def start_free_spin(self, state: RoundState, ledger: EventLedger) -> RoundState:
        if state.mode != self.config.freegame_type or state.free_spins_remaining <= 0:
            raise ValueError("no free spin is available")
        next_state = replace(
            state,
            spin_index=state.spin_index + 1,
            cascade_index=0,
            free_spins_remaining=state.free_spins_remaining - 1,
        )
        ledger.free_spin_start(
            next_state.spin_index,
            next_state.free_spins_remaining,
            next_state.column_levels,
            next_state.stage,
            next_state.departures,
        )
        return next_state

    def retrigger(
        self,
        state: RoundState,
        ledger: EventLedger,
        board: Board,
        scatter_positions: Sequence[int],
    ) -> RoundState:
        if state.mode != self.config.freegame_type:
            raise ValueError("retrigger can only occur in freegame")
        positions = validate_trigger_positions(board, scatter_positions, self.config)
        next_state = replace(
            state,
            free_spins_remaining=(
                state.free_spins_remaining + self.config.retrigger_free_spins
            ),
        )
        ledger.retrigger(
            positions,
            self.config.retrigger_free_spins,
            next_state.free_spins_remaining,
            next_state.column_levels,
            next_state.stage,
            next_state.departures,
            board,
        )
        return next_state

    def resolve_departures(
        self,
        state: RoundState,
        ledger: EventLedger,
        groups: Sequence[DepartureGroup],
    ) -> RoundState:
        levels = list(state.column_levels)
        departures = state.departures
        selection_stage = (
            "yard"
            if state.mode == self.config.basegame_type
            else stage_for_departures(departures)
        )
        if any(group.stage_at_selection != selection_stage for group in groups):
            raise ValueError("departure stageAtSelection does not match frozen evaluation state")
        for group in groups:
            reset_level = 0 if state.mode == self.config.basegame_type else RESET_LEVELS[group.stage_at_selection]
            for column in group.columns:
                levels[column] = reset_level
            departures += len(group.columns)
            stage_after = (
                "yard"
                if state.mode == self.config.basegame_type
                else stage_for_departures(departures)
            )
            ledger.departure_resolve(
                group,
                [reset_level] * len(group.columns),
                departures if state.mode == self.config.freegame_type else 0,
                stage_after,
            )
        return replace(
            state,
            column_levels=tuple(levels),
            departures=departures if state.mode == self.config.freegame_type else 0,
            stage=(
                stage_for_departures(departures)
                if state.mode == self.config.freegame_type
                else "yard"
            ),
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )

    def complete_round(self, state: RoundState, ledger: EventLedger) -> RoundState:
        if state.capped != ledger.capped:
            raise ValueError("state and ledger cap status do not match")
        final_levels = (0, 0, 0, 0, 0, 0)
        ledger.round_complete(final_levels)
        self.active_outcome_path = "natural"
        return RoundState(
            mode=self.config.basegame_type,
            column_levels=final_levels,
            round_payout_units=ledger.round_payout_units,
            capped=ledger.capped,
        )

    def complete_bonus(
        self, state: RoundState, ledger: EventLedger, feature_payout_units: int
    ) -> RoundState:
        if state.mode != self.config.freegame_type:
            raise ValueError("bonus completion requires freegame state")
        if state.capped != ledger.capped:
            raise ValueError("state and ledger cap status do not match")
        if ledger.capped:
            return self.complete_round(state, ledger)
        if state.free_spins_remaining != 0:
            raise ValueError("bonus completion requires zero remaining spins")
        if feature_payout_units != ledger.feature_payout_units:
            raise ValueError("bonus payout must be ledger-derived")
        ledger.bonus_complete(ledger.feature_payout_units)
        return self.complete_round(state, ledger)
